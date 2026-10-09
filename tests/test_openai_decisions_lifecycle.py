"""Exercise the real OpenAI SDK's transport and resource ownership without paid calls."""

from __future__ import annotations

import asyncio
import json
import traceback

import pytest

openai = pytest.importorskip("openai", minversion="3.26.0")
httpx2 = pytest.importorskip("httpx2")

from typed_evals import (  # noqa: E402
    EvaluationSample,
    Evaluator,
    GuardPolicy,
    GuardrailViolation,
    OpenAIDecisionsBackend,
    RuntimeGuard,
    ToolAccuracy,
    ToolProposal,
)
from typed_evals.errors import OpenAIDecisionsError  # noqa: E402

API_KEY = "test-only-not-a-real-openai-key"


def sdk_client(handler, *, retries=0, timeout=2):
    return openai.AsyncOpenAI(
        api_key=API_KEY,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=retries,
        timeout=timeout,
    )


def wire_response(answers):
    return {
        "model": "gpt-6-luna",
        "answers": answers,
        "usage": {
            "input_tokens": 25,
            "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 1},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 27,
        },
    }


def predicate_answer(name="answer_relevancy", probability=0.9):
    return {"name": name, "type": "predicate", "probability": probability}


def failure_response(status, message="temporary provider failure"):
    return httpx2.Response(
        status,
        json={"error": {"type": "api_error", "message": message}},
        # Keep genuine SDK retries deterministic and fast without replacing the retry loop.
        headers={"retry-after-ms": "1"},
    )


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_sdk_retries_transient_status_without_second_retry_loop(status, sample):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/v1/decisions"
        if len(calls) == 1:
            return failure_response(status)
        return httpx2.Response(200, json=wire_response([predicate_answer()]))

    async with sdk_client(handler, retries=1) as client:
        result = await Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate_one(
            sample
        )
        assert not client.is_closed()
    assert len(calls) == 2
    assert result.metrics["answer_relevancy"].raw_score == 0.9


@pytest.mark.parametrize(
    "status,error_type",
    [
        (400, "BadRequestError"),
        (401, "AuthenticationError"),
        (403, "PermissionDeniedError"),
        (422, "UnprocessableEntityError"),
    ],
)
async def test_permanent_failures_are_not_retried(status, error_type, sample):
    calls = []

    def handler(request):
        calls.append(request)
        return failure_response(status, "permanent provider failure")

    async with sdk_client(handler, retries=2) as client:
        with pytest.raises(OpenAIDecisionsError, match=error_type):
            await Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate([sample])
        assert not client.is_closed()
    assert len(calls) == 1


async def test_retry_exhaustion_records_safe_error_without_provider_content(sample):
    calls = []

    def handler(request):
        calls.append(request)
        return failure_response(503, f"private provider response: {API_KEY}")

    async with sdk_client(handler, retries=2) as client:
        backend = OpenAIDecisionsBackend(api_key=API_KEY, client=client)
        report = await Evaluator(backend=backend, errors="record").aevaluate([sample])
        assert not client.is_closed()
    assert len(calls) == 3
    row = report.results[0]
    assert row.metrics["answer_relevancy"].status == "error"
    assert row.metrics["answer_relevancy"].score is None
    assert row.passed is None
    exported = json.dumps(report.to_dict())
    assert "private provider response" not in exported
    assert API_KEY not in exported
    assert API_KEY not in repr(backend)


async def test_raised_provider_failure_keeps_credentials_out_of_public_exception(sample):
    def handler(request):
        return failure_response(401, f"private provider payload echoed credentials: {API_KEY}")

    async with sdk_client(handler) as client:
        with pytest.raises(OpenAIDecisionsError, match="AuthenticationError") as caught:
            await Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate([sample])
        assert not client.is_closed()
    error = caught.value
    public_error = "\n".join((str(error), repr(error), "".join(traceback.format_exception(error))))
    assert API_KEY not in public_error
    assert "private provider payload" not in public_error
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


async def test_sdk_timeout_retry_count_and_recorded_unavailability(sample):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx2.ReadTimeout("private timeout detail", request=request)

    async with sdk_client(handler, retries=1) as client:
        with pytest.raises(OpenAIDecisionsError, match="APITimeoutError"):
            await Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate([sample])
        assert len(calls) == 2
        report = await Evaluator(
            backend=OpenAIDecisionsBackend(client=client), errors="record"
        ).aevaluate([sample])
        assert not client.is_closed()
    assert len(calls) == 4
    assert report.results[0].passed is None
    assert "private timeout detail" not in json.dumps(report.to_dict())


async def test_injected_client_configuration_and_ownership_are_preserved(sample):
    calls = []

    def handler(request):
        calls.append(request)
        return failure_response(503)

    async with sdk_client(handler, retries=0, timeout=0.127) as client:
        original_timeout = client.timeout
        original_retries = client.max_retries
        backend = OpenAIDecisionsBackend(client=client, timeout=41, max_retries=7)
        await Evaluator(backend=backend, errors="record").aevaluate([sample])
        assert client.timeout == original_timeout
        assert client.max_retries == original_retries == 0
        assert calls[0].extensions["timeout"]["read"] == 0.127
        assert len(calls) == 1
        assert not client.is_closed()
    assert client.is_closed()


async def test_cancelling_judgment_leaves_injected_client_open(sample):
    entered = asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(request)
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled transport must not return")

    async with sdk_client(handler, retries=2) as client:
        task = asyncio.create_task(
            Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate([sample])
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.is_closed()
    assert len(requests) == 1


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
async def test_owned_client_closes_on_success_failure_and_cancellation(
    outcome, sample, monkeypatch
):
    clients, constructors = [], []
    actual_client_class = openai.AsyncOpenAI
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        if outcome == "cancelled":
            await asyncio.Event().wait()
        if outcome == "error":
            return failure_response(400)
        return httpx2.Response(200, json=wire_response([predicate_answer()]))

    def factory(**kwargs):
        constructors.append(kwargs)
        client = actual_client_class(
            **kwargs,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    evaluator = Evaluator(
        backend=OpenAIDecisionsBackend(api_key=API_KEY, timeout=13.5, max_retries=0)
    )
    task = asyncio.create_task(evaluator.aevaluate([sample]))
    if outcome == "cancelled":
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(OpenAIDecisionsError, match="BadRequestError"):
            await task
    else:
        assert (await task).results[0].passed is True
    assert len(clients) == 1
    assert clients[0].is_closed()
    assert constructors[0]["timeout"] == 13.5
    assert constructors[0]["max_retries"] == 0


async def test_owned_client_is_shared_across_workers_and_created_per_batch(monkeypatch):
    actual_client_class = openai.AsyncOpenAI
    clients, calls = [], []
    active = peak = 0
    three_workers_entered = asyncio.Event()

    async def handler(request):
        nonlocal active, peak
        calls.append(json.loads(request.content))
        active += 1
        peak = max(peak, active)
        if active == 3:
            three_workers_entered.set()
        try:
            await asyncio.wait_for(three_workers_entered.wait(), timeout=2)
            await asyncio.sleep(0)
            return httpx2.Response(200, json=wire_response([predicate_answer()]))
        finally:
            active -= 1

    def factory(**kwargs):
        client = actual_client_class(
            **kwargs,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    evaluator = Evaluator(
        backend=OpenAIDecisionsBackend(api_key=API_KEY, max_retries=0), max_concurrency=3
    )
    samples = [EvaluationSample(input=str(index), response="A") for index in range(9)]
    first = await evaluator.aevaluate(samples)
    assert len(first.results) == 9
    assert len(clients) == 1
    assert clients[0].is_closed()
    assert peak == 3
    assert active == 0
    assert len(calls) == 9
    await evaluator.aevaluate(samples[:1])
    assert len(clients) == 2
    assert clients[1].is_closed()
    assert clients[1] is not clients[0]


async def test_owned_client_uses_normal_openai_environment_key(monkeypatch, sample):
    actual_client_class = openai.AsyncOpenAI
    requests, clients = [], []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=wire_response([predicate_answer()]))

    def factory(**kwargs):
        client = actual_client_class(
            **kwargs,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    backend = OpenAIDecisionsBackend()
    result = await Evaluator(backend=backend).aevaluate_one(sample)
    assert result.passed is True
    assert requests[0].headers["Authorization"] == f"Bearer {API_KEY}"
    assert clients[0].is_closed()
    assert API_KEY not in repr(backend)


@pytest.mark.parametrize("model", [None, True, 42, "", "  \n"])
def test_invalid_model_configuration_is_rejected(model):
    with pytest.raises(ValueError, match="model"):
        OpenAIDecisionsBackend(model=model)


@pytest.mark.parametrize("timeout", [True, False, None, "30", 0, -1, float("inf"), float("nan")])
def test_invalid_timeout_configuration_is_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        OpenAIDecisionsBackend(timeout=timeout)


@pytest.mark.parametrize("max_retries", [True, False, None, "2", -1, 1.0])
def test_invalid_retry_configuration_is_rejected(max_retries):
    with pytest.raises(ValueError, match="max_retries"):
        OpenAIDecisionsBackend(max_retries=max_retries)


@pytest.mark.parametrize("injected", [True, False])
async def test_sdk_without_decisions_capability_explains_installation_and_client_ownership(
    injected, monkeypatch
):
    actual_client_class = openai.AsyncOpenAI
    clients = []

    def unsupported_decisions(client):
        raise AttributeError("SDK has no Decisions resource")

    def handler(request):
        raise AssertionError("unsupported SDK must fail before an HTTP request")

    def factory(**kwargs):
        client = actual_client_class(
            **kwargs,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        clients.append(client)
        return client

    # Simulate an older SDK's absent resource on a genuine SDK client.
    monkeypatch.setattr(actual_client_class, "decisions", property(unsupported_decisions))
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    supplied_client = factory(api_key=API_KEY) if injected else None
    backend = OpenAIDecisionsBackend(api_key=API_KEY, client=supplied_client)
    try:
        with pytest.raises(ImportError, match=r"SDK >=3\.26\.0.*typed_evals\[openai\]") as caught:
            async with backend.session():
                raise AssertionError("unsupported SDK must not produce a judge session")
        assert API_KEY not in str(caught.value)
        assert len(clients) == 1
        assert clients[0].is_closed() is (not injected)
    finally:
        if supplied_client is not None:
            await supplied_client.close()


@pytest.mark.parametrize("errors", ["raise", "record"])
@pytest.mark.parametrize("unavailable", ["refusal", "missing", "invalid"])
async def test_actual_runtime_tool_gate_blocks_unavailable_decisions(errors, unavailable):
    calls, requests = [], []

    def handler(request):
        requests.append(json.loads(request.content))
        if unavailable == "refusal":
            answers = [{"name": "tool_accuracy", "type": "refusal"}]
        elif unavailable == "missing":
            answers = []
        else:
            answers = [
                {
                    "name": "tool_accuracy",
                    "type": "choice",
                    "choice": "true",
                    "confidence": 0.9,
                    "probabilities": [
                        {"value": "true", "probability": 0.9},
                        {"value": "false", "probability": "0.1"},
                    ],
                }
            ]
        return httpx2.Response(200, json=wire_response(answers))

    sample = EvaluationSample(
        input="Look up item 42",
        response="",
        contexts=["lookup(item_id: int) retrieves an item."],
        proposed_tool_call=ToolProposal(name="lookup", arguments={"item_id": 42}),
        metadata={"private": "not judge evidence"},
    )
    async with sdk_client(handler) as client:
        guard = RuntimeGuard(
            {
                "before_tool": GuardPolicy(
                    Evaluator(
                        [ToolAccuracy()],
                        backend=OpenAIDecisionsBackend(client=client),
                        errors=errors,
                    )
                )
            }
        )
        with pytest.raises(GuardrailViolation) as caught:
            await guard.acall_tool({"lookup": lambda item_id: calls.append(item_id)}, sample)
        assert not client.is_closed()
    assert not calls
    decision = caught.value.decision
    assert not decision.allowed
    assert decision.action == "block"
    if errors == "record":
        assert decision.unavailable_metrics == ("tool_accuracy",)
        assert decision.evaluation.passed is None
        assert decision.evaluation.metrics["tool_accuracy"].score is None
        assert decision.evaluation.metrics["tool_accuracy"].passed is None
    elif unavailable == "refusal":
        assert "DecisionRefusalError" in decision.error
    request = requests[0]
    assert request["questions"][0]["name"] == "tool_accuracy"
    assert request["questions"][0]["type"] == "choice"
    evidence = json.loads(request["input"])
    assert evidence["proposed_tool_call"] == {"name": "lookup", "arguments": {"item_id": 42}}
    assert "metadata" not in evidence
    assert "trace" not in evidence
