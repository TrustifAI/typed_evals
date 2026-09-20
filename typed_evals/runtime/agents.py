"""Decorate native agents without importing or mutating an orchestration framework."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

from typed_evals.data.models import EvaluationSample
from typed_evals.runtime.guards import RuntimeGuard, _async_callable, guarded_by

_ENTRYPOINTS = (
    "invoke",
    "ainvoke",
    "run",
    "arun",
    "kickoff",
    "kickoff_async",
    "chat",
    "achat",
    "__call__",
)
_STREAMING = frozenset(
    {"stream", "astream", "run_stream", "run_stream_sync", "astream_events", "invoke_stream"}
)


class GuardedAgent:
    """Proxy for explicitly guarded entrypoints; native helper attributes are forwarded.

    The underlying agent is never patched. Internal method calls therefore retain their
    native return types. Register guarded tools/middleware to intercept internal actions.
    """

    __slots__ = ("_agent", "_methods")

    def __init__(self, agent: Any, methods: dict[str, Callable]) -> None:
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_methods", methods)

    @property
    def __wrapped__(self) -> Any:
        return self._agent

    @property
    def guarded_methods(self) -> tuple[str, ...]:
        return tuple(self._methods)

    def __getattr__(self, name: str) -> Any:
        if name in self._methods:
            return self._methods[name]
        if name in _STREAMING:
            raise AttributeError("Streaming entrypoints must be buffered behind a guard")
        if name in _ENTRYPOINTS:
            raise AttributeError(f"Agent entrypoint {name!r} was not selected for guarding")
        return getattr(self._agent, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if "__call__" not in self._methods:
            raise TypeError("This agent has no guarded __call__; use one of guarded_methods")
        return self._methods["__call__"](*args, **kwargs)


def guarded_agent(
    guard: RuntimeGuard,
    *,
    before: str | None = None,
    after: str | None = None,
    before_sample: Callable[[tuple, dict], EvaluationSample] | None = None,
    after_sample: Callable[[Any, tuple, dict], EvaluationSample] | None = None,
    methods: Sequence[str] | None = None,
    async_methods: Sequence[str] = (),
    factory: bool = False,
) -> Callable:
    """Decorate an agent instance, class constructor, factory, or invocation function.

    Common non-streaming entrypoints are discovered, or supplied explicitly with methods.
    Evidence mappers receive public call arguments (never self/cls) and native output.
    async_methods marks regular methods returning awaitables, e.g. Agent.run in some SDKs.
    Classes and factory=True functions construct GuardedAgent proxies without judging init.
    """
    options = dict(
        before=before, after=after, before_sample=before_sample, after_sample=after_sample
    )
    # Validate policy names and builders before constructing or calling anything.
    wrap_default = guarded_by(guard, **options)
    selected = None if methods is None else _method_names(methods, "methods")
    forced_async = _method_names(async_methods, "async_methods", allow_empty=True)
    if type(factory) is not bool:
        raise TypeError("factory must be a boolean")
    if selected is not None and set(forced_async) - set(selected):
        raise ValueError("async_methods must be included in methods")
    if set(selected or ()) & _STREAMING:
        raise ValueError("Streaming entrypoints must be materialized before guarding")

    def wrap_instance(agent: Any) -> GuardedAgent:
        names = (
            selected
            if selected is not None
            else tuple(name for name in _ENTRYPOINTS if callable(getattr(agent, name, None)))
        )
        if not names:
            raise TypeError("No agent entrypoints found; provide methods=('your_method',)")
        if set(forced_async) - set(names):
            raise ValueError("async_methods contains an unavailable agent method")
        wrapped = {}
        for name in names:
            method = getattr(agent, name, None)
            if not callable(method):
                raise TypeError(f"Agent method {name!r} is missing or not callable")
            decorate = (
                guarded_by(guard, **options, async_mode=True)
                if name in forced_async
                else wrap_default
            )
            wrapped[name] = _reject_streaming(decorate(method))
        return GuardedAgent(agent, wrapped)

    def decorate(target: Any) -> Any:
        if inspect.isclass(target) or factory:
            if not callable(target):
                raise TypeError("Agent factory must be callable")
            if not inspect.isclass(target) and _async_callable(target):

                @wraps(target)
                async def async_factory(*args: Any, **kwargs: Any) -> GuardedAgent:
                    return wrap_instance(await target(*args, **kwargs))

                return async_factory

            @wraps(target)
            def constructor(*args: Any, **kwargs: Any) -> GuardedAgent:
                return wrap_instance(target(*args, **kwargs))

            return constructor
        if inspect.isroutine(target):
            if selected is not None and selected != ("__call__",):
                raise ValueError(
                    "Function entrypoint is __call__; use factory=True for agent factories"
                )
            if set(forced_async) - {"__call__"}:
                raise ValueError("Function async_methods may only contain __call__")
            decorate_function = (
                guarded_by(guard, **options, async_mode=True) if forced_async else wrap_default
            )
            return _reject_streaming(decorate_function(target))
        return wrap_instance(target)

    return decorate


def _method_names(
    value: Sequence[str], label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of method names, not one string")
    names = tuple(value)
    if (not names and not allow_empty) or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError(f"{label} must contain nonempty method names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate method names")
    if any(
        (name.startswith("__") and name != "__call__")
        or name in {"_agent", "_methods", "guarded_methods"}
        for name in names
    ):
        raise ValueError(f"{label} contains a reserved proxy attribute")
    return names


def _reject_streaming(function: Callable) -> Callable:
    # Some agents combine streaming and non-streaming into the same entrypoint.
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        if kwargs.get("stream") is True or bound.arguments.get("stream") is True:
            raise TypeError("Use a materialized, non-streaming agent call before evaluation")
        return function(*args, **kwargs)

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await wrapper(*args, **kwargs)

        return async_wrapper
    return wrapper
