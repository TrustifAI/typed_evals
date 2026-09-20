"""Create native CrewAI tools whose functions are evaluated before execution."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, get_type_hints

from typed_evals.backends import Backend
from typed_evals.data.models import ToolProposal
from typed_evals.runtime.guards import FailureAction
from typed_evals.runtime.tools import guard_tool as python_guard_tool

if TYPE_CHECKING:
    from crewai.tools import BaseTool


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
) -> Callable[[Callable], BaseTool]:
    """Decorate a Python function to produce a guarded native CrewAI Tool.

    Use this in place of crewai.tools.tool, then register the result with the agent.
    input/contexts callbacks receive bound JSON arguments, including defaults.
    Capture request-specific evidence in a tool factory; model-generated
    arguments are not authenticated runtime context.

    Successful results are not cached by this tool. Use Crew(cache=False) when
    guarding calls: a previously populated shared cache can bypass tool execution.
    A direct run/arun raises GuardrailViolation on block; CrewAI's agent loop may
    report it as a tool error and continue. No global hooks are registered.
    """
    try:
        from crewai.tools import tool
        from crewai.tools.base_tool import Tool
        from crewai.utilities.string_utils import sanitize_tool_name
    except ImportError as exc:
        raise ImportError("Install CrewAI support: pip install 'typed_evals[crewai]'") from exc

    class GuardedTool(Tool):
        async def _arun(self, *args, **kwargs):
            if inspect.iscoroutinefunction(self.func):
                return await self.func(*args, **kwargs)
            return await asyncio.to_thread(self.func, *args, **kwargs)

        def to_structured_tool(self):
            structured = super().to_structured_tool()
            # Preserve coroutine detection; CrewAI's sync _run proxy hides it.
            structured.func = self.func
            return structured

    def decorate(function: Callable) -> BaseTool:
        # CrewAI publishes sanitized names; judge the same name the agent sees.
        raw_name = function.__name__ if name is None else name
        ToolProposal(name=raw_name)
        tool_name = sanitize_tool_name(raw_name)
        wrapped = python_guard_tool(
            policy=policy,
            input=input,
            contexts=contexts,
            threshold=threshold,
            backend=backend,
            on_fail=on_fail,
            on_error=on_error,
            timeout=timeout,
            name=tool_name,
        )(function)
        signature = inspect.signature(function)
        if any(p.kind == p.POSITIONAL_ONLY for p in signature.parameters.values()):
            raise TypeError("CrewAI tools require parameters that can be passed by keyword")
        # CrewAI inspects the signature, so resolve postponed annotations there too.
        hints = get_type_hints(function, include_extras=True)
        wrapped.__signature__ = signature.replace(
            parameters=[
                parameter.replace(annotation=hints.get(key, parameter.annotation))
                for key, parameter in signature.parameters.items()
            ],
            return_annotation=hints.get("return", signature.return_annotation),
        )
        registered = tool(tool_name)(wrapped)
        return GuardedTool(
            name=registered.name,
            description=inspect.getdoc(function),
            args_schema=registered.args_schema,
            result_schema=registered.result_schema,
            func=wrapped,
            cache_function=_never_cache,
        )

    return decorate


def _never_cache(*args: Any, **kwargs: Any) -> bool:
    return False
