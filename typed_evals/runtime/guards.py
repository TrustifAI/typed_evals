"""Explicit, framework-independent checkpoints inside an application execution loop."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import Field, JsonValue

from typed_evals._utils import run_sync
from typed_evals.data.models import EvaluationSample, Model, SampleResult, ToolCall
from typed_evals.errors import CalibrationError, InvalidAnswerError, TypedEvalsError
from typed_evals.evaluation.decorators import ResponseEvaluator

FailureAction = Literal["annotate", "retry", "escalate", "block"]
Action = Literal["allow", "annotate", "retry", "escalate", "block"]
_PRIORITY = {"allow": 0, "annotate": 1, "retry": 2, "escalate": 3, "block": 4}
T = TypeVar("T")


@dataclass
class _DecisionLog:
    decisions: list[GuardDecision] = field(default_factory=list)
    active: bool = True


_decision_log: ContextVar[_DecisionLog | None] = ContextVar("jev_decision_log", default=None)


@contextmanager
def _decision_scope():
    parent = _decision_log.get()
    current = _DecisionLog()
    token = _decision_log.set(current)
    try:
        yield current.decisions
    finally:
        current.active = False
        _decision_log.reset(token)
        if parent is not None and parent.active:
            parent.decisions.extend(current.decisions)


def _record_decision(decision: GuardDecision) -> None:
    log = _decision_log.get()
    if log is not None and log.active:
        log.decisions.append(decision)


@dataclass(frozen=True)
class GuardPolicy:
    """One evaluator (or fitted pipeline) and the disposition of its metric results.

    Retry and escalation are instructions for the host, never automatic side effects.
    `on_error` also covers missing evidence, skipped metrics and judge timeouts.
    """

    evaluator: ResponseEvaluator
    on_fail: FailureAction = "block"
    on_error: FailureAction = "block"
    metric_actions: Mapping[str, FailureAction] = field(default_factory=dict)
    timeout: float | None = None

    def __post_init__(self) -> None:
        actions = dict(self.metric_actions)
        for action in (self.on_fail, self.on_error, *actions.values()):
            if not isinstance(action, str) or action not in _PRIORITY or action == "allow":
                raise ValueError("failure actions must be block, annotate, retry, or escalate")
        if any(not isinstance(name, str) or not name.strip() for name in actions):
            raise ValueError("metric_actions keys must be nonempty metric names")
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        if not callable(getattr(self.evaluator, "aevaluate_one", None)):
            raise TypeError("evaluator must implement aevaluate_one")
        object.__setattr__(self, "metric_actions", MappingProxyType(actions))


class GuardDecision(Model):
    """Serializable audit information. `allowed` is policy, not a quality score."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    checkpoint: str
    action: Action
    evaluation: SampleResult | None = None
    failed_metrics: tuple[str, ...] = ()
    unavailable_metrics: tuple[str, ...] = ()
    error: str | None = None

    @property
    def allowed(self) -> bool:
        return self.action in ("allow", "annotate")

    @property
    def hallucination_suspected(self) -> bool | None:
        if self.evaluation is None:
            return None
        grounding = [
            metric
            for name, metric in self.evaluation.metrics.items()
            if name in ("faithfulness", "tool_grounding")
        ]
        if any(
            metric.status == "ok" and metric.score is not None and metric.passed is False
            for metric in grounding
        ):
            return True
        if grounding and all(
            metric.status == "ok" and metric.score is not None and metric.passed is True
            for metric in grounding
        ):
            return False
        return None

    def require_allowed(self) -> None:
        if not self.allowed:
            raise GuardrailViolation(self)

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.update(allowed=self.allowed, hallucination_suspected=self.hallucination_suspected)
        if self.evaluation is not None:
            data["evaluation"]["passed"] = self.evaluation.passed
            for name, metric in self.evaluation.metrics.items():
                data["evaluation"]["metrics"][name]["score"] = metric.score
        return data


class GuardrailViolation(TypedEvalsError):
    """The host must block, revise/recheck, or obtain approval before continuing."""

    def __init__(self, decision: GuardDecision) -> None:
        self.decision = decision
        super().__init__(f"Checkpoint {decision.checkpoint!r}: {decision.action}")


@dataclass(frozen=True)
class GuardedResponse(Generic[T]):
    """Native output plus per-invocation decisions; no shared conversation state."""

    output: T
    decisions: tuple[GuardDecision, ...]

    @property
    def metadata(self) -> dict[str, Any]:
        flags = [decision.hallucination_suspected for decision in self.decisions]
        # An unavailable grounding check keeps the aggregate unknown unless another failed.
        relevant = [
            decision.hallucination_suspected
            for decision in self.decisions
            if decision.evaluation is not None
            and {"faithfulness", "tool_grounding"} & decision.evaluation.metrics.keys()
        ]
        # On a failed request the evaluator may not report which metrics were unavailable.
        if any(decision.error is not None for decision in self.decisions):
            relevant.append(None)
        suspected = (
            True
            if True in flags
            else (False if relevant and all(v is False for v in relevant) else None)
        )
        return {
            "jev": {
                "decisions": [decision.to_dict() for decision in self.decisions],
                "hallucination_suspected": suspected,
            }
        }


class RuntimeGuard:
    """Evaluate named checkpoints with explicit enforcement and native-output wrappers.

    Names are application-defined: before_tool, after_response, before_memory_write, etc.
    Unconfigured checkpoints raise instead of silently allowing an unchecked action.
    """

    def __init__(self, policies: Mapping[str, GuardPolicy]) -> None:
        if not policies or any(
            not isinstance(name, str) or not name.strip() or not isinstance(policy, GuardPolicy)
            for name, policy in policies.items()
        ):
            raise ValueError("policies must map nonempty checkpoint names to GuardPolicy objects")
        self.policies = MappingProxyType(dict(policies))

    def _policy(self, checkpoint: str) -> GuardPolicy:
        if checkpoint not in self.policies:
            raise ValueError(f"No guard policy configured for checkpoint {checkpoint!r}")
        return self.policies[checkpoint]

    def check(self, checkpoint: str, sample: EvaluationSample) -> GuardDecision:
        """Return a decision without executing an action. Use enforce/run to enforce it."""
        return run_sync(lambda: self.acheck(checkpoint, sample))

    async def acheck(self, checkpoint: str, sample: EvaluationSample) -> GuardDecision:
        policy = self._policy(checkpoint)
        snapshot = _snapshot(sample)
        try:
            request = policy.evaluator.aevaluate_one(snapshot)
            result = (
                await request
                if policy.timeout is None
                else await asyncio.wait_for(request, timeout=policy.timeout)
            )
            if not isinstance(result, SampleResult) or not result.metrics:
                raise InvalidAnswerError("Evaluator returned no metric results")
        except CalibrationError:
            # A broken calibration configuration is always fatal, as in EvaluationPipeline.
            raise
        except Exception as exc:
            # Cancellation/KeyboardInterrupt propagate. Never export provider exception text.
            return GuardDecision(
                checkpoint=checkpoint,
                action=policy.on_error,
                error=f"Evaluation failed: {type(exc).__name__}",
            )
        unknown = policy.metric_actions.keys() - result.metrics.keys()
        if unknown:
            raise ValueError(f"metric_actions contains unknown metrics: {sorted(unknown)}")
        failed = tuple(
            name
            for name, metric in result.metrics.items()
            if metric.status == "ok" and metric.passed is False
        )
        unavailable = tuple(
            name
            for name, metric in result.metrics.items()
            if metric.status != "ok" or metric.passed is None or metric.score is None
        )
        actions = [policy.metric_actions.get(name, policy.on_fail) for name in failed]
        if unavailable:
            actions.append(policy.on_error)
        action = max(actions, key=_PRIORITY.__getitem__) if actions else "allow"
        return GuardDecision(
            checkpoint=checkpoint,
            action=action,
            evaluation=result,
            failed_metrics=failed,
            unavailable_metrics=unavailable,
        )

    def enforce(self, checkpoint: str, sample: EvaluationSample) -> GuardDecision:
        decision = self.check(checkpoint, sample)
        _record_decision(decision)
        decision.require_allowed()
        return decision

    async def aenforce(self, checkpoint: str, sample: EvaluationSample) -> GuardDecision:
        decision = await self.acheck(checkpoint, sample)
        _record_decision(decision)
        decision.require_allowed()
        return decision

    def respond(self, checkpoint: str, sample: EvaluationSample, output: T) -> GuardedResponse[T]:
        """Evaluate a materialized native response before returning it to its consumer."""
        _materialized(output)
        return GuardedResponse(output, (self.enforce(checkpoint, sample),))

    async def arespond(
        self, checkpoint: str, sample: EvaluationSample, output: T
    ) -> GuardedResponse[T]:
        _materialized(output)
        return GuardedResponse(output, (await self.aenforce(checkpoint, sample),))

    def run(
        self,
        checkpoint: str,
        sample: EvaluationSample,
        operation: Callable[[], T],
        *,
        after: str | None = None,
        sample_builder: Callable[[T, EvaluationSample], EvaluationSample] | None = None,
    ) -> GuardedResponse[T]:
        """Check, then call a lazy operation once. Optional post-check gates its output.

        The caller must bind the operation to the checked evidence. For tool dispatch with
        argument snapshots, use call_tool. A post-check cannot undo an operation's effects.
        """
        self._validate_operation(operation, after, sample_builder)
        if _async_callable(operation):
            raise TypeError("Use arun for async operations")
        snapshot = _snapshot(sample)
        decisions = [self.enforce(checkpoint, snapshot)]
        output = operation()
        _materialized(output)
        if after is not None:
            decisions.append(self.enforce(after, sample_builder(output, snapshot)))
        return GuardedResponse(output, tuple(decisions))

    async def arun(
        self,
        checkpoint: str,
        sample: EvaluationSample,
        operation: Callable[[], Any],
        *,
        after: str | None = None,
        sample_builder: Callable[[Any, EvaluationSample], EvaluationSample] | None = None,
    ) -> GuardedResponse:
        """Async run; synchronous operations run in a worker to keep the loop responsive."""
        self._validate_operation(operation, after, sample_builder)
        snapshot = _snapshot(sample)
        decisions = [await self.aenforce(checkpoint, snapshot)]
        output = await _invoke(operation)
        _materialized(output)
        if after is not None:
            decisions.append(await self.aenforce(after, sample_builder(output, snapshot)))
        return GuardedResponse(output, tuple(decisions))

    def _validate_operation(
        self, operation: Callable, after: str | None, builder: Callable | None
    ) -> None:
        _ordinary_callable(operation)
        if (after is None) != (builder is None):
            raise ValueError("after and sample_builder must be provided together")
        if after is not None:
            self._policy(after)
            _sync_mapper(builder)

    def call_tool(
        self,
        tools: Mapping[str, Callable],
        sample: EvaluationSample,
        *,
        before: str = "before_tool",
        after: str | None = None,
        output_mapper: Callable[[Any], JsonValue] | None = None,
    ) -> GuardedResponse:
        """Dispatch the exact proposed name and a private snapshot of its JSON arguments.

        Tools accept keyword arguments. The optional post-check sees an observed ToolCall;
        non-JSON outputs need an explicit output_mapper. Tool exceptions propagate unchanged.
        """
        snapshot, function, arguments = _prepare_tool(tools, sample)
        if _async_callable(function):
            raise TypeError("Use acall_tool for async tools")
        if output_mapper is not None:
            _sync_mapper(output_mapper)
        builder = _tool_result_builder(output_mapper) if after is not None else None
        return self.run(
            before,
            snapshot,
            lambda: function(**arguments),
            after=after,
            sample_builder=builder,
        )

    async def acall_tool(
        self,
        tools: Mapping[str, Callable],
        sample: EvaluationSample,
        *,
        before: str = "before_tool",
        after: str | None = None,
        output_mapper: Callable[[Any], JsonValue] | None = None,
    ) -> GuardedResponse:
        snapshot, function, arguments = _prepare_tool(tools, sample)
        if output_mapper is not None:
            _sync_mapper(output_mapper)

        async def invoke() -> Any:
            return await _invoke(function, **arguments)

        builder = _tool_result_builder(output_mapper) if after is not None else None
        return await self.arun(before, snapshot, invoke, after=after, sample_builder=builder)


def guarded_by(
    guard: RuntimeGuard,
    *,
    before: str | None = None,
    after: str | None = None,
    before_sample: Callable[[tuple, dict], EvaluationSample] | None = None,
    after_sample: Callable[[Any, tuple, dict], EvaluationSample] | None = None,
    native_output: bool = False,
    async_mode: bool | None = None,
) -> Callable:
    """Wrap sync/async framework calls using explicit synchronous evidence mappers.

    Returns GuardedResponse, including nested enforced decisions. Set native_output=True
    for registered tools: their return type stays intact and an enclosing decorated agent
    collects their decisions. async_mode=True supports ordinary methods returning awaitables.
    An after-only wrapper cannot prevent internal effects. Use call_tool to bind arguments.
    """
    if before is None and after is None:
        raise ValueError("At least one checkpoint is required")
    if type(native_output) is not bool or (async_mode is not None and type(async_mode) is not bool):
        raise TypeError("native_output and async_mode must be booleans")
    for checkpoint, builder in ((before, before_sample), (after, after_sample)):
        if (checkpoint is None) != (builder is None):
            raise ValueError("Each checkpoint requires its corresponding sample builder")
        if checkpoint is not None:
            guard._policy(checkpoint)
            _sync_mapper(builder)

    def decorator(function: Callable) -> Callable:
        _ordinary_callable(function)
        if async_mode is False and _async_callable(function):
            raise TypeError("async_mode=False cannot wrap an asynchronous callable")
        if async_mode is True or _async_callable(function):

            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _decision_scope() as decisions:
                    if before is not None:
                        await guard.aenforce(before, before_sample(args, kwargs))
                    output = function(*args, **kwargs)
                    if inspect.isawaitable(output):
                        output = await output
                    _materialized(output)
                    if after is not None:
                        await guard.aenforce(after, after_sample(output, args, kwargs))
                    return output if native_output else GuardedResponse(output, tuple(decisions))

            return async_wrapper

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _decision_scope() as decisions:
                if before is not None:
                    guard.enforce(before, before_sample(args, kwargs))
                output = function(*args, **kwargs)
                _materialized(output)
                if after is not None:
                    guard.enforce(after, after_sample(output, args, kwargs))
                return output if native_output else GuardedResponse(output, tuple(decisions))

        return wrapper

    return decorator


def _snapshot(sample: EvaluationSample) -> EvaluationSample:
    if not isinstance(sample, EvaluationSample):
        raise TypeError("Pass an EvaluationSample built from the current workflow state")
    # Frozen models still contain mutable dicts. Revalidate and copy at the trust boundary.
    return EvaluationSample.model_validate(deepcopy(sample.model_dump(mode="python")))


def _async_callable(function: Callable) -> bool:
    return inspect.iscoroutinefunction(function) or (
        callable(function) and inspect.iscoroutinefunction(function.__call__)
    )


def _ordinary_callable(function: Callable) -> None:
    if not callable(function):
        raise TypeError("operation must be a lazy callable")
    for candidate in (function, function.__call__):
        if inspect.isgeneratorfunction(candidate) or inspect.isasyncgenfunction(candidate):
            raise TypeError("Streaming functions must be materialized before evaluation")


def _sync_mapper(builder: Callable) -> None:
    _ordinary_callable(builder)
    if _async_callable(builder):
        raise TypeError("sample builders and output mappers must be synchronous")


def _materialized(output: Any) -> None:
    if inspect.isawaitable(output) or inspect.isgenerator(output) or inspect.isasyncgen(output):
        if inspect.iscoroutine(output) or inspect.isgenerator(output):
            output.close()
        raise TypeError("Await asynchronous output and materialize streams before evaluation")


async def _invoke(function: Callable, *args: Any, **kwargs: Any) -> Any:
    if _async_callable(function):
        return await function(*args, **kwargs)
    output = await asyncio.to_thread(function, *args, **kwargs)
    return await output if inspect.isawaitable(output) else output


def _prepare_tool(tools: Mapping[str, Callable], sample: EvaluationSample) -> tuple:
    snapshot = _snapshot(sample)
    proposal = snapshot.proposed_tool_call
    if proposal is None:
        raise ValueError("call_tool requires sample.proposed_tool_call")
    if proposal.name not in tools:
        raise ValueError(f"No registered tool named {proposal.name!r}")
    function = tools[proposal.name]
    _ordinary_callable(function)
    arguments = deepcopy(proposal.arguments)
    # Catch misspelled/missing arguments before making a judge request or touching the tool.
    inspect.signature(function).bind(**arguments)
    return snapshot, function, arguments


def _tool_result_builder(mapper: Callable[[Any], JsonValue] | None) -> Callable:
    def build(output: Any, sample: EvaluationSample) -> EvaluationSample:
        proposal = sample.proposed_tool_call
        event = ToolCall(
            name=proposal.name,
            arguments=deepcopy(proposal.arguments),
            output=mapper(output) if mapper is not None else output,
            status="success",
        )
        return EvaluationSample.model_validate(
            {
                **sample.model_dump(mode="python"),
                "trace": (*sample.trace, event),
                "proposed_tool_call": None,
            }
        )

    return build
