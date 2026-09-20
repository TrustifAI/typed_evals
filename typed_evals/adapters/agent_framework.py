"""Microsoft Agent Framework tools with an injected FunctionInvocationContext."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from typed_evals.backends import Backend
from typed_evals.runtime.guards import FailureAction, _sync_mapper
from typed_evals.runtime.tools import _contexts, _decorate_tool, _make_guard


def guard_tool(
    *,
    policy: str,
    input: str | Callable[[Any], str],
    contexts: Sequence[str] | Callable[[Any], Sequence[str]],
    threshold: float = 0.9,
    backend: Backend | None = None,
    on_fail: FailureAction = "block",
    on_error: FailureAction = "block",
    timeout: float | None = None,
    name: str | None = None,
    context_parameter: str = "ctx",
) -> Callable:
    """Apply below agent_framework.tool to a sync/async function with a context.

    Annotate the injected parameter as FunctionInvocationContext. input/contexts
    callbacks receive that context; use ctx.kwargs for application-owned per-run
    evidence supplied through function_invocation_kwargs. Runtime objects, session
    state, and unrelated kwargs are excluded from the judged tool arguments.
    If @tool uses a custom name, pass the same name here.

    A blocked direct invocation raises GuardrailViolation. An agent's automatic
    tool loop may convert this into a tool-error result and continue its run.
    """
    if not isinstance(context_parameter, str) or not context_parameter.isidentifier():
        raise ValueError("context_parameter must be a Python parameter name")
    for value in (input, contexts):
        if callable(value):
            _sync_mapper(value)
    fixed_contexts = None if callable(contexts) else _contexts(contexts)
    guard = _make_guard(policy, threshold, backend, on_fail, on_error, timeout)

    def evidence(arguments):
        ctx = arguments[context_parameter]
        return {
            "input": input(ctx) if callable(input) else input,
            "contexts": contexts(ctx) if callable(contexts) else fixed_contexts,
            "id": ctx.metadata.get("call_id"),
        }

    def decorate(function):
        return _decorate_tool(function, guard, evidence, name=name, injected=(context_parameter,))

    return decorate
