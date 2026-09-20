from __future__ import annotations

import asyncio

import pytest
from conftest import FakeBackend

pytest.importorskip("crewai")

from crewai.tools import BaseTool

from typed_evals import GuardrailViolation
from typed_evals.adapters.crewai import guard_tool


@pytest.mark.parametrize("async_tool", [False, True])
@pytest.mark.parametrize("dispatch", ["run", "arun", "invoke", "ainvoke"])
@pytest.mark.parametrize("score", [0.99, 0.1])
async def test_native_and_structured_tools_enforce_before_execution(async_tool, dispatch, score):
    backend = FakeBackend(
        lambda state, questions: {name: {"type": "noul", "noul": score} for name in questions}
    )
    executed = []
    output = {"text": "Refund approved"}

    def read(ticket_id: str, full: bool = False) -> dict:
        """Read an owned ticket."""
        executed.append((ticket_id, full))
        return output

    async def aread(ticket_id: str, full: bool = False) -> dict:
        """Read an owned ticket asynchronously."""
        return read(ticket_id, full)

    registered = guard_tool(
        policy="Read owned tickets only",
        input=lambda args: f"Read {args['ticket_id']}",
        contexts=["Alice owns T-42"],
        backend=backend,
        name="Read Ticket",
    )(aread if async_tool else read)
    assert isinstance(registered, BaseTool)
    schema = registered.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"ticket_id", "full"}
    assert schema["properties"]["full"]["default"] is False
    assert registered.cache_function({}, output) is False
    structured = registered.to_structured_tool()
    assert structured.cache_function({}, output) is False

    async def invoke():
        if dispatch == "run":
            return await asyncio.to_thread(registered.run, ticket_id="T-42")
        if dispatch == "arun":
            return await registered.arun(ticket_id="T-42")
        if dispatch == "invoke":
            return await asyncio.to_thread(structured.invoke, {"ticket_id": "T-42"})
        return await structured.ainvoke({"ticket_id": "T-42"})

    if score < 0.9:
        with pytest.raises(GuardrailViolation):
            await invoke()
        assert not executed
    else:
        assert await invoke() is output
        assert executed == [("T-42", False)]
    assert backend.calls[0][0]["input"] == "Read T-42"
    assert backend.calls[0][0]["proposed_tool_call"] == {
        "name": structured.name,
        "arguments": {"ticket_id": "T-42", "full": False},
    }
    assert backend.sessions == backend.closed == 1


@pytest.mark.parametrize("mode", ["exception", "timeout", "missing_answer"])
async def test_unavailable_judge_blocks_crewai_dispatch(mode):
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

    @guard_tool(
        policy="Owned only",
        input="Read",
        contexts=["Owned"],
        backend=backend,
        timeout=0.01 if mode == "timeout" else None,
    )
    async def read(ticket_id: str) -> str:
        """Read an owned ticket."""
        executed.append(ticket_id)
        return ticket_id

    with pytest.raises(GuardrailViolation):
        await read.to_structured_tool().ainvoke({"ticket_id": "T-42"})
    assert not executed
    assert backend.sessions == backend.closed == 1


async def test_crewai_preserves_reviewed_arguments_when_evidence_callback_mutates_them():
    backend = FakeBackend()
    executed = []

    def evidence(args):
        args["payload"]["ids"].append("T-99")
        return ["Alice owns T-42"]

    @guard_tool(policy="Owned only", input="Read", contexts=evidence, backend=backend)
    async def read(payload: dict) -> dict:
        """Read owned tickets."""
        executed.append(payload)
        return payload

    payload = {"ids": ["T-42"]}
    with pytest.raises(GuardrailViolation):
        # FakeBackend's default score is below the tool safety threshold.
        await read.arun(payload=payload)
    backend.answer = lambda state, questions: {
        name: {"type": "noul", "noul": 0.99} for name in questions
    }
    assert await read.to_structured_tool().ainvoke({"payload": payload}) == payload
    assert executed == [{"ids": ["T-42"]}]
    assert payload == {"ids": ["T-42"]}


def test_crewai_schema_rejects_invalid_input_before_judging():
    backend = FakeBackend()

    @guard_tool(policy="Owned only", input="Read", contexts=["Owned"], backend=backend)
    def read(ticket_id: str) -> str:
        """Read an owned ticket."""
        raise AssertionError("must not execute")

    with pytest.raises(ValueError, match="validation"):
        read.to_structured_tool().invoke({"ticket_id": []})
    assert not backend.calls


def test_crewai_rejects_positional_only_parameters():
    def read(ticket_id: str, /) -> str:
        """Read a ticket."""
        return ticket_id

    with pytest.raises(TypeError, match="keyword"):
        guard_tool(policy="Owned only", input="Read", contexts=["Owned"])(read)
