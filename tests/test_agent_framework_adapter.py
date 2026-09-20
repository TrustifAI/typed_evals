from __future__ import annotations

import asyncio

import pytest
from conftest import FakeBackend

pytest.importorskip("agent_framework")

from agent_framework import FunctionInvocationContext, tool

from typed_evals import GuardrailViolation
from typed_evals.adapters.agent_framework import guard_tool


def runtime(function, **kwargs):
    return FunctionInvocationContext(
        function=function,
        arguments={},
        kwargs={
            "request": "Read T-42",
            "evidence": ["Alice owns T-42"],
            "secret": "private",
            **kwargs,
        },
        metadata={"call_id": "call-42"},
    )


@pytest.mark.parametrize("async_tool", [False, True])
@pytest.mark.parametrize("score", [0.99, 0.1])
async def test_native_framework_injects_context_and_guards_normalized_arguments(async_tool, score):
    backend = FakeBackend(
        lambda state, questions: {name: {"type": "noul", "noul": score} for name in questions}
    )
    executed = []
    output = {"text": "Refund approved"}

    def read(ticket_id: str, ctx: FunctionInvocationContext, full: bool = False) -> dict:
        """Read an owned ticket."""
        assert ctx.kwargs["secret"] == "private"
        executed.append((ticket_id, full))
        return output

    async def aread(ticket_id: str, ctx: FunctionInvocationContext, full: bool = False) -> dict:
        """Read an owned ticket asynchronously."""
        return read(ticket_id, ctx, full)

    wrapped = guard_tool(
        policy="Read owned tickets only",
        input=lambda ctx: ctx.kwargs["request"],
        contexts=lambda ctx: ctx.kwargs["evidence"],
        backend=backend,
        name="read_ticket",
    )(aread if async_tool else read)
    registered = tool(name="read_ticket")(wrapped)
    assert set(registered.parameters()["properties"]) == {"ticket_id", "full"}
    context = runtime(registered)
    if score < 0.9:
        with pytest.raises(GuardrailViolation) as exc:
            await registered.invoke(arguments={"ticket_id": "T-42"}, context=context)
        assert exc.value.decision.evaluation.sample_id == "call-42"
        assert not executed
    else:
        result = await registered.invoke(
            arguments={"ticket_id": "T-42"}, context=context, skip_parsing=True
        )
        assert result is output
        assert executed == [("T-42", False)]
    judged = backend.calls[0][0]
    assert judged["input"] == "Read T-42"
    assert judged["proposed_tool_call"] == {
        "name": "read_ticket",
        "arguments": {"ticket_id": "T-42", "full": False},
    }
    assert "private" not in str(judged)
    assert "FunctionInvocationContext" not in judged["contexts"][0]
    assert backend.sessions == backend.closed == 1


@pytest.mark.parametrize("mode", ["exception", "timeout", "missing_answer"])
async def test_framework_unavailable_judgments_block_without_execution(mode):
    backend = FakeBackend()
    if mode == "exception":

        def fail(*args):
            raise RuntimeError("private failure")

        backend.answer = fail
    elif mode == "timeout":
        backend.delay = 0.1
    else:
        backend.answer = lambda *args: {}
    executed = []

    @tool
    @guard_tool(
        policy="Owned only",
        input="Read",
        contexts=["Owned"],
        backend=backend,
        timeout=0.01 if mode == "timeout" else None,
    )
    async def read(ticket_id: str, ctx: FunctionInvocationContext) -> str:
        """Read an owned ticket."""
        executed.append(ticket_id)
        return ticket_id

    with pytest.raises(GuardrailViolation):
        await read.invoke(arguments={"ticket_id": "T-42"}, context=runtime(read))
    assert not executed
    assert backend.sessions == backend.closed == 1


async def test_framework_missing_evidence_does_not_judge_or_execute():
    backend = FakeBackend()

    @tool
    @guard_tool(
        policy="Owned only",
        input=lambda ctx: ctx.kwargs["request"],
        contexts=lambda ctx: ctx.kwargs["evidence"],
        backend=backend,
    )
    def read(ticket_id: str, ctx: FunctionInvocationContext) -> str:
        """Read a ticket."""
        raise AssertionError("must not execute")

    with pytest.raises(ValueError, match="contexts"):
        await read.invoke(arguments={"ticket_id": "T-42"}, context=runtime(read, evidence=[]))
    assert not backend.calls


async def test_framework_alternate_context_parameter_and_argument_snapshot():
    backend = FakeBackend(
        lambda state, questions: {name: {"type": "noul", "noul": 0.99} for name in questions},
        delay=0.02,
    )
    executed = []

    @tool
    @guard_tool(
        policy="Owned only",
        input="Read",
        contexts=["Alice owns T-42"],
        backend=backend,
        context_parameter="invocation",
    )
    async def read(payload: dict, invocation: FunctionInvocationContext) -> dict:
        """Read owned tickets."""
        executed.append(payload)
        return payload

    context = runtime(read)
    task = asyncio.create_task(
        read.invoke(arguments={"payload": {"ids": ["T-42"]}}, context=context, skip_parsing=True)
    )
    while not backend.calls:
        await asyncio.sleep(0)
    context.arguments["payload"]["ids"].append("T-99")
    assert await task == {"ids": ["T-42"]}
    assert executed == [{"ids": ["T-42"]}]
    assert "invocation" not in read.parameters()["properties"]


def test_framework_requires_declared_context_parameter():
    def read(ticket_id: str) -> str:
        return ticket_id

    with pytest.raises(ValueError, match="injected parameters"):
        guard_tool(policy="Owned only", input="Read", contexts=["Owned"])(read)


def test_framework_rejects_async_evidence_callbacks():
    async def evidence(ctx):
        return ["Owned"]

    with pytest.raises(TypeError, match="synchronous"):
        guard_tool(policy="Owned only", input="Read", contexts=evidence)
