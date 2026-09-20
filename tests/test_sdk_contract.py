"""Exercise the actual typesafe-sdk, with only its HTTP boundary mocked."""

import json

import httpx2
import pytest
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
)

from typed_evals import (
    AnswerRelevancy,
    EvaluationSample,
    Evaluator,
    Faithfulness,
    GuardPolicy,
    GuardrailViolation,
    JevBackend,
    Metric,
    RuntimeGuard,
    ToolAccuracy,
    ToolProposal,
)


def sdk_client(handler, retries=0):
    return AsyncTypeSafeClient(
        api_key="test-only-not-a-real-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=retries, backoff_initial=0, backoff_max=0),
        timeout=2,
    )


def wire_response(answers):
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 25, "output_tokens": 0},
        "answers": answers,
    }


async def test_request_serialization_and_all_primitive_responses(sample):
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        assert request.url.path == "/v1/systemone"
        return httpx2.Response(
            200,
            json=wire_response(
                {
                    "faithfulness": {"type": "noul", "noul": 0.9},
                    "verdict": {
                        "type": "choice",
                        "choice": "fail",
                        "confidence": 0.6,
                        "probabilities": {"pass": 0.2, "fail": 0.8},
                    },
                    "coverage": {
                        "type": "score",
                        "score": 1.5,
                        "confidence": 0.4,
                        "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                        "legend": {"0": "none", "1": "some", "2": "all"},
                    },
                }
            ),
        )

    metrics = [
        Faithfulness(),
        Metric(
            name="verdict",
            kind="choice",
            instructions="Does response pass?",
            criteria={"pass": "acceptable", "fail": "unacceptable"},
            pass_options=["pass"],
            pass_definition="Response is acceptable",
        ),
        Metric(
            name="coverage",
            kind="score",
            instructions="How much does response cover?",
            criteria=["none", "some", "all"],
            pass_definition="All essentials are covered",
        ),
    ]
    async with sdk_client(handler) as client:
        report = await Evaluator(metrics, backend=JevBackend(client=client)).aevaluate([sample])
        # Evaluator must not close a caller-owned SDK client.
        await client.system_one(state="test", questions={"test": AnswerRelevancy().question()})
    body = observed[0]
    assert set(body) == {"state", "questions", "model"}
    assert set(body["questions"]) == {"faithfulness", "verdict", "coverage"}
    assert body["questions"]["faithfulness"]["type"] == "noul"
    assert body["model"] == "jev-1.13.0"
    row = report.results[0]
    assert row.metrics["faithfulness"].confidence is None
    assert row.metrics["verdict"].raw_score == 0.2
    assert row.metrics["coverage"].raw_score == 0.75
    assert row.usage["input_tokens"] == 25


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_sdk_retries_transient_status_without_nested_retry_loop(status, sample):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(status, json={"error": {"message": "temporary"}})
        return httpx2.Response(
            200, json=wire_response({"answer_relevancy": {"type": "noul", "noul": 0.7}})
        )

    async with sdk_client(handler, retries=1) as client:
        result = await Evaluator(backend=JevBackend(client=client)).aevaluate_one(sample)
    assert result.metrics["answer_relevancy"].score == 0.7
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_sdk_does_not_retry_bad_requests_or_authentication(status, sample):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(status, json={"error": {"message": "rejected"}})

    async with sdk_client(handler, retries=2) as client:
        with pytest.raises(TypeSafeAPIError):
            await Evaluator(backend=JevBackend(client=client)).aevaluate([sample])
    assert len(calls) == 1


async def test_real_sdk_rejects_malformed_wire_response(sample):
    def handler(request):
        return httpx2.Response(
            200, json=wire_response({"answer_relevancy": {"type": "noul", "noul": "bad"}})
        )

    async with sdk_client(handler) as client:
        with pytest.raises(TypeSafeAPIResponseValidationError):
            await Evaluator(backend=JevBackend(client=client)).aevaluate([sample])


async def test_retry_exhaustion_can_be_recorded(sample):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(503, json={"error": {"message": "private provider response"}})

    async with sdk_client(handler, retries=1) as client:
        report = await Evaluator(backend=JevBackend(client=client), errors="record").aevaluate(
            [sample]
        )
    assert len(calls) == 2
    assert report.results[0].metrics["answer_relevancy"].status == "error"
    assert "private" not in str(report.to_dict())


@pytest.mark.parametrize("score,executed", [(0.1, False), (0.79, False), (0.8, True), (0.95, True)])
async def test_runtime_tool_gate_through_real_sdk_serialization(score, executed):
    requests, calls = [], []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=wire_response(
                {
                    "tool_accuracy": {
                        "type": "choice",
                        "choice": "true" if score >= 0.5 else "false",
                        "confidence": 0.6,
                        "probabilities": {"true": score, "false": 1 - score},
                    }
                }
            ),
        )

    sample = EvaluationSample(
        input="Look up item 42",
        response="",
        contexts=("lookup(item_id: int) retrieves an item.",),
        proposed_tool_call=ToolProposal(name="lookup", arguments={"item_id": 42}),
        metadata={"private": "not judge evidence"},
    )
    async with sdk_client(handler) as client:
        guard = RuntimeGuard(
            {
                "before_tool": GuardPolicy(
                    Evaluator([ToolAccuracy()], backend=JevBackend(client=client))
                )
            }
        )
        operation = guard.acall_tool({"lookup": lambda item_id: calls.append(item_id)}, sample)
        if executed:
            result = await operation
            decision = result.decisions[0]
        else:
            with pytest.raises(GuardrailViolation) as caught:
                await operation
            decision = caught.value.decision
    assert decision.evaluation.tool_name == "lookup"
    assert decision.to_dict()["evaluation"]["tool_name"] == "lookup"
    metric = decision.evaluation.metrics["tool_accuracy"]
    assert metric.raw_score == score
    assert metric.confidence == 0.6
    assert metric.selected_choice == ("true" if score >= 0.5 else "false")
    assert calls == ([42] if executed else [])
    question = requests[0]["questions"]["tool_accuracy"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"true", "false"}
    state = requests[0]["state"]
    # The SDK sends the evidence object intact, including the unexecuted proposal.
    assert state["proposed_tool_call"] == {"name": "lookup", "arguments": {"item_id": 42}}
    assert "trace" not in state
    assert "metadata" not in state
