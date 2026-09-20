"""Small tool decorators backed by the same runtime enforcement as explicit guards."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import wraps
from typing import Any, get_type_hints
from uuid import uuid4

from typed_evals.backends import Backend
from typed_evals.data.models import EvaluationSample, ToolProposal
from typed_evals.evaluation.evaluator import Evaluator
from typed_evals.metrics import ToolSafety
from typed_evals.runtime.guards import (
    FailureAction,
    GuardPolicy,
    RuntimeGuard,
    _async_callable,
    _materialized,
    _ordinary_callable,
    _sync_mapper,
)


def guard_tool(
    *,
    policy: str,
    input: str | Callable[[Mapping[str, Any]], str],
    contexts: Sequence[str] | Callable[[Mapping[str, Any]], Sequence[str]],
    threshold: float = 0.9,
    backend: Backend | None = None,
    on_fail: FailureAction = "block",
    on_error: FailureAction = "block",
    timeout: float | None = None,
    name: str | None = None,
) -> Callable:
    """Guard a sync/async Python tool, preserving its signature and native output.

    input/contexts can be fixed values or synchronous functions of bound arguments.
    Supply application-owned authorization facts in contexts. Captures the tool name,
    JSON arguments (including defaults), signature/docstring, and a unique call ID.
    Register the decorated function with your framework. For injected runtime objects,
    use a framework adapter or the advanced guarded_by API.
    """
    for value in (input, contexts):
        if callable(value):
            _sync_mapper(value)
    fixed_contexts = None if callable(contexts) else _contexts(contexts)
    guard = _make_guard(policy, threshold, backend, on_fail, on_error, timeout)

    def evidence(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": input(arguments) if callable(input) else input,
            "contexts": contexts(arguments) if callable(contexts) else fixed_contexts,
        }

    def decorate(function: Callable) -> Callable:
        return _decorate_tool(function, guard, evidence, name=name)

    return decorate


def _make_guard(policy, threshold, backend, on_fail, on_error, timeout) -> RuntimeGuard:
    return RuntimeGuard(
        {
            "before_tool": GuardPolicy(
                Evaluator([ToolSafety(policy=policy, threshold=threshold)], backend=backend),
                on_fail=on_fail,
                on_error=on_error,
                timeout=timeout,
            )
        }
    )


def _contexts(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(
            "contexts must be a nonempty sequence of application-owned evidence strings"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("contexts must contain nonblank strings")
    return tuple(value)


def _decorate_tool(
    function: Callable,
    guard: RuntimeGuard,
    evidence: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    name: str | None = None,
    injected: tuple[str, ...] = (),
) -> Callable:
    _ordinary_callable(function)
    signature = inspect.signature(function)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
        raise TypeError("guard_tool requires named parameters; use guarded_by for variadic tools")
    if set(injected) - signature.parameters.keys():
        raise ValueError(f"Tool must declare injected parameters: {', '.join(injected)}")
    tool_name = function.__name__ if name is None else name
    ToolProposal(name=tool_name)  # Validate even before a first invocation.
    public_signature = signature.replace(
        parameters=[p for p in signature.parameters.values() if p.name not in injected]
    )
    description = f"{tool_name}{public_signature}\n{inspect.getdoc(function) or ''}".strip()

    def prepare(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        runtime = {key: bound.arguments[key] for key in injected}
        # Copy before callbacks or awaits. The tool receives the exact JSON arguments judged,
        # even if the caller or an evidence callback mutates the original nested objects.
        arguments = deepcopy({k: v for k, v in bound.arguments.items() if k not in injected})
        proposal = ToolProposal(name=tool_name, arguments=arguments)
        facts = evidence({**deepcopy(proposal.arguments), **runtime})
        sample = EvaluationSample(
            input=facts["input"],
            response="",
            contexts=(description, *_contexts(facts["contexts"])),
            id=facts.get("id") or uuid4().hex,
            proposed_tool_call=proposal.model_copy(deep=True),
        )
        execution = inspect.BoundArguments(signature, {**proposal.arguments, **runtime})
        return sample, execution

    if _async_callable(function):

        @wraps(function)
        async def async_wrapper(*args, **kwargs):
            sample, execution = prepare(args, kwargs)
            await guard.aenforce("before_tool", sample)
            output = await function(*execution.args, **execution.kwargs)
            _materialized(output)
            return output

        wrapper = async_wrapper
    else:

        @wraps(function)
        def sync_wrapper(*args, **kwargs):
            sample, execution = prepare(args, kwargs)
            guard.enforce("before_tool", sample)
            output = function(*execution.args, **execution.kwargs)
            _materialized(output)
            return output

        wrapper = sync_wrapper
    # Resolve postponed annotations in their defining module for framework schema generation.
    wrapper.__annotations__ = get_type_hints(function, include_extras=True)
    return wrapper
