"""LangChain ToolRuntime adapter. Apply below @tool, above the Python function."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from typed_evals.backends import Backend
from typed_evals.runtime.guards import FailureAction, _sync_mapper
from typed_evals.runtime.tools import _contexts, _decorate_tool, _make_guard


def guard_tool(
    *,
    policy: str,
    contexts: Sequence[str] | Callable[[Any], Sequence[str]],
    threshold: float = 0.9,
    backend: Backend | None = None,
    on_fail: FailureAction = "block",
    on_error: FailureAction = "block",
    timeout: float | None = None,
    name: str | None = None,
) -> Callable:
    """Guard a LangChain tool declaring runtime: ToolRuntime.

    Captures the latest human text, tool call ID, name and bound JSON arguments.
    contexts is application-owned evidence or a synchronous callback taking ToolRuntime.
    Runtime objects, state, config and credentials are never automatically sent to Jev.
    Match name to @tool(name) when registering under a different name.
    """
    if callable(contexts):
        _sync_mapper(contexts)
    fixed_contexts = None if callable(contexts) else _contexts(contexts)
    guard = _make_guard(policy, threshold, backend, on_fail, on_error, timeout)

    def evidence(arguments):
        runtime = arguments["runtime"]
        return {
            "input": _latest_request(runtime.state),
            "id": runtime.tool_call_id,
            "contexts": contexts(runtime) if callable(contexts) else fixed_contexts,
        }

    def decorate(function):
        return _decorate_tool(function, guard, evidence, name=name, injected=("runtime",))

    return decorate


def _latest_request(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("messages", ())):
        if isinstance(message, Mapping):
            role = message.get("role", message.get("type"))
            content = message.get("content")
        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", None)
        if role not in ("human", "user"):
            continue
        if isinstance(content, str) and content.strip():
            return content
        # Text blocks are supported, but silently dropping image/audio evidence is not.
        if (
            isinstance(content, list)
            and content
            and all(
                isinstance(block, Mapping)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                for block in content
            )
        ):
            text = "\n".join(block["text"] for block in content)
            if text.strip():
                return text
        raise ValueError(
            "guard_tool requires a nonempty text request; use guarded_by for custom evidence"
        )
    raise ValueError("guard_tool requires a human message in runtime.state['messages']")
