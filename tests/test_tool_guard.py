from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from conftest import FakeBackend

from typed_evals import (
    EvaluationSample,
    Evaluator,
    GuardPolicy,
    GuardrailViolation,
    RuntimeGuard,
    guard_tool,
    guarded_by,
)
from typed_evals.adapters.langchain import _latest_request
from typed_evals.adapters.langchain import guard_tool as langchain_guard_tool


def judge(score=0.99):
    return FakeBackend(
        lambda state, questions: {name: {"type": "noul", "noul": score} for name in questions}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("score,allowed", [(0.99, True), (0.1, False)])
async def test_tool_decorator_binds_defaults_preserves_output_and_blocks(
    score, allowed, asynchronous
):
    backend = judge(score)
    executed = []
    native = {"result": "T-42"}

    def lookup(ticket_id: str, /, *, full: bool = False) -> dict:
        """Look up an owned ticket."""
        executed.append((ticket_id, full))
        return native

    async def async_lookup(ticket_id: str, /, *, full: bool = False) -> dict:
        return lookup(ticket_id, full=full)

    original = async_lookup if asynchronous else lookup
    wrapped = guard_tool(
        policy="Read owned tickets only.",
        input=lambda args: f"Read {args['ticket_id']}",
        contexts=lambda args: ["Alice owns T-42."],
        backend=backend,
    )(original)
    assert inspect.signature(wrapped) == inspect.signature(original)
    assert wrapped.__name__ == original.__name__
    if allowed:
        result = await wrapped("T-42") if asynchronous else wrapped("T-42")
        assert result is native
        assert executed == [("T-42", False)]
    else:
        with pytest.raises(GuardrailViolation) as exc:
            if asynchronous:
                await wrapped("T-42")
            else:
                wrapped("T-42")
        assert exc.value.decision.failed_metrics == ("tool_safety",)
        assert not executed
    state = backend.calls[0][0]
    assert state["input"] == "Read T-42"
    assert state["proposed_tool_call"] == {
        "name": original.__name__,
        "arguments": {"ticket_id": "T-42", "full": False},
    }
    assert state["contexts"][-1] == "Alice owns T-42."


async def test_mutations_during_review_or_evidence_mapping_cannot_change_dispatch():
    backend = judge()
    backend.delay = 0.03
    original = {"ids": ["T-42"]}
    executed = []

    def contexts(args):
        args["payload"]["ids"].append("T-99")
        return ["Alice owns T-42."]

    @guard_tool(policy="Owned only", input="Read my ticket", contexts=contexts, backend=backend)
    async def read(payload):
        executed.append(payload)

    task = asyncio.create_task(read(original))
    while not backend.calls:
        await asyncio.sleep(0)
    original["ids"].append("T-100")
    await task
    assert executed == [{"ids": ["T-42"]}]
    assert backend.calls[0][0]["proposed_tool_call"]["arguments"] == {"payload": {"ids": ["T-42"]}}


@pytest.mark.parametrize("mode", ["exception", "timeout", "missing_answer"])
async def test_unavailable_judgments_block_without_execution(mode):
    backend = judge()
    if mode == "exception":

        def fail(*args):
            raise RuntimeError("private provider error")

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
    async def read(ticket_id):
        executed.append(ticket_id)

    with pytest.raises(GuardrailViolation):
        await read("T-42")
    assert not executed
    assert backend.closed == 1


def test_native_tool_exceptions_are_not_retried():
    backend = judge()
    executed = []

    @guard_tool(policy="Owned only", input="Read", contexts=["Owned"], backend=backend)
    def read(ticket_id):
        executed.append(ticket_id)
        raise LookupError("not found")

    with pytest.raises(LookupError, match="not found"):
        read("T-42")
    assert executed == ["T-42"]
    assert len(backend.calls) == 1


async def test_cancellation_during_judgment_never_executes_tool():
    backend = judge()
    backend.delay = 10
    executed = []

    @guard_tool(policy="Owned only", input="Read", contexts=["Owned"], backend=backend)
    async def read(ticket_id):
        executed.append(ticket_id)

    task = asyncio.create_task(read("T-42"))
    while not backend.calls:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not executed
    assert backend.closed == 1


def test_explicit_annotation_allows_a_failed_judgment():
    backend = judge(0.1)

    @guard_tool(
        policy="Owned only", input="Read", contexts=["Owned"], backend=backend, on_fail="annotate"
    )
    def read(ticket_id):
        return ticket_id

    assert read("T-42") == "T-42"


def test_missing_dynamic_evidence_prevents_judging_and_execution():
    backend = judge()

    @guard_tool(policy="Owned only", input="Read", contexts=lambda args: [], backend=backend)
    def read(ticket_id):
        raise AssertionError("must not run")

    with pytest.raises(ValueError, match="contexts"):
        read("T-42")
    assert backend.sessions == 0


def test_decisions_and_unique_ids_reach_enclosing_agent():
    backend = judge()

    @guard_tool(policy="Owned only", input="Read", contexts=["Owned"], backend=backend)
    def read(ticket_id):
        return ticket_id

    guard = RuntimeGuard({"after": GuardPolicy(Evaluator(backend=backend))})

    @guarded_by(
        guard, after="after", after_sample=lambda *args: EvaluationSample(input="Q", response="A")
    )
    def agent():
        return [read("T-42"), read("T-42")]

    result = agent()
    assert result.output == ["T-42", "T-42"]
    assert [d.checkpoint for d in result.decisions] == ["before_tool", "before_tool", "after"]
    assert result.decisions[0].evaluation.sample_id != result.decisions[1].evaluation.sample_id


@pytest.mark.parametrize("contexts", [[], "Owned", [""]])
def test_invalid_static_evidence_rejected_early(contexts):
    with pytest.raises(ValueError, match="contexts"):
        guard_tool(policy="Owned only", input="Read", contexts=contexts)


def test_invalid_arguments_do_not_reach_judge():
    backend = judge()

    @guard_tool(policy="Owned only", input="Read", contexts=["Owned"], backend=backend)
    def read(ticket_id):
        raise AssertionError("must not run")

    with pytest.raises(TypeError):
        read(typo="T-42")
    assert backend.sessions == 0


def test_langchain_adapter_reads_latest_request_and_excludes_runtime():
    backend = judge()
    runtime = SimpleNamespace(
        state={
            "messages": [
                {"role": "user", "content": "old"},
                {"role": "user", "content": "Read T-42"},
                {"role": "assistant", "content": "calling tool"},
            ]
        },
        tool_call_id="call-42",
        context={"authorization": "Alice owns T-42", "secret": "private"},
    )

    @langchain_guard_tool(
        policy="Owned only", contexts=lambda r: [r.context["authorization"]], backend=backend
    )
    def read(ticket_id, runtime):
        assert runtime.context["secret"] == "private"
        return ticket_id

    assert read(ticket_id="T-42", runtime=runtime) == "T-42"
    assert "private" not in str(backend.calls)
    assert "runtime" not in backend.calls[0][0]["contexts"][0]
    assert backend.calls[0][0]["input"] == "Read T-42"
    assert backend.calls[0][0]["proposed_tool_call"]["arguments"] == {"ticket_id": "T-42"}


@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "user", "content": ""}], [{"role": "user", "content": [{"type": "image"}]}]],
)
def test_langchain_adapter_never_substitutes_stale_or_missing_human_evidence(messages):
    with pytest.raises(ValueError, match="request|human message"):
        _latest_request({"messages": messages})


def test_langchain_text_blocks():
    assert (
        _latest_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Read"},
                            {"type": "text", "text": "T-42"},
                        ],
                    }
                ]
            }
        )
        == "Read\nT-42"
    )
