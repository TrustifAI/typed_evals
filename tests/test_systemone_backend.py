"""System One wire compatibility without provider SDK coercion or network calls."""

from __future__ import annotations

import asyncio
import json
import traceback

import httpx2
import pytest
from conftest import calibrated_rows

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    GuardPolicy,
    GuardrailViolation,
    ImageInput,
    Metric,
    RuntimeGuard,
    SystemOneBackend,
)
from typed_evals.backends import systemone
from typed_evals.backends.provenance import backend_provenance
from typed_evals.errors import (
    CalibrationMismatchError,
    InvalidAnswerError,
    SystemOneError,
    UnsupportedModalityError,
)

MODEL = "open-laya"
BASE_URL = "https://judge.example/prefix"
API_KEY = "test-only-private-key"


def backend(client=None, **kwargs):
    return SystemOneBackend(
        model=MODEL,
        base_url=BASE_URL,
        client=client,
        max_retries=kwargs.pop("max_retries", 0),
        **kwargs,
    )


def wire_response(answers=None, **kwargs):
    return {
        "model": MODEL,
        "answers": answers
        if answers is not None
        else {"answer_relevancy": {"type": "noul", "noul": 0.9}},
        **kwargs,
    }


def panel():
    return [
        Metric(name="predicate", instructions="Pass?", pass_definition="Pass"),
        Metric(
            name="classification",
            kind="choice",
            instructions="Classify",
            criteria={"good": "Good", "bad": "Bad"},
            pass_options=("good",),
            pass_definition="Good",
        ),
        Metric(
            name="ordinal",
            kind="score",
            instructions="Rate",
            criteria=("Poor", "Fair", "Good"),
            pass_definition="Good",
        ),
    ]


async def test_roundtrip_all_primitives_preserves_sdk_questions_and_selected_evidence(sample):
    requests = []
    answers = {
        "predicate": {"type": "noul", "noul": 0.9},
        "classification": {
            "type": "choice",
            "choice": "good",
            "confidence": 0.8,
            "probabilities": {"good": 0.8, "bad": 0.2},
        },
        "ordinal": {
            "type": "score",
            "score": 1.5,
            "confidence": 0.75,
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
        },
    }

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            200, json=wire_response(answers, usage={"input_tokens": 12, "output_tokens": 0})
        )

    metrics = panel()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        result = await Evaluator(metrics, backend=backend(client)).aevaluate_one(
            sample.model_copy(update={"metadata": {"private": "omit"}})
        )
        assert not client.is_closed
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/prefix/v1/systemone"
    payload = json.loads(request.content)
    assert payload == {
        "model": MODEL,
        "state": {"input": sample.input, "response": sample.response},
        "questions": {metric.name: metric.question().model_dump(mode="json") for metric in metrics},
    }
    assert result.model == MODEL
    assert result.metrics["predicate"].raw_score == 0.9
    assert result.metrics["classification"].raw_score == 0.8
    assert result.metrics["classification"].selected_choice == "good"
    assert result.metrics["ordinal"].raw_score == 0.75
    assert result.metrics["ordinal"].raw_kind == "normalized_ordinal_score"
    assert result.usage == {"input_tokens": 12, "output_tokens": 0}


@pytest.mark.parametrize("api_key", [None, API_KEY])
async def test_authentication_is_explicit_and_ignores_typesafe_environment(
    api_key, monkeypatch, sample
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "unrelated-typesafe-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=wire_response())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        await Evaluator(backend=backend(client, api_key=api_key)).aevaluate_one(sample)
    assert requests[0].headers.get("authorization") == (f"Bearer {API_KEY}" if api_key else None)
    assert "unrelated-typesafe-secret" not in requests[0].content.decode()


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
async def test_owned_client_is_pooled_and_closed_on_every_exit(outcome, monkeypatch, sample):
    actual_client = httpx2.AsyncClient
    clients, requests = [], []
    entered = asyncio.Event()

    async def handler(request):
        requests.append(request)
        if outcome == "cancelled":
            entered.set()
            await asyncio.Event().wait()
        if outcome == "error":
            return httpx2.Response(401, text="private provider failure")
        return httpx2.Response(200, json=wire_response())

    def factory(**kwargs):
        client = actual_client(**kwargs, transport=httpx2.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(systemone.httpx2, "AsyncClient", factory)
    evaluator = Evaluator(backend=backend(), max_concurrency=2)
    if outcome == "cancelled":
        task = asyncio.create_task(evaluator.aevaluate([sample]))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(SystemOneError):
            await evaluator.aevaluate([sample])
    else:
        await evaluator.aevaluate([sample, sample])
        assert len(requests) == 2
    assert len(clients) == 1
    assert clients[0].is_closed


async def test_cancelled_request_leaves_caller_client_open_and_does_not_retry(sample):
    entered = asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(request)
        entered.set()
        await asyncio.Event().wait()

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        task = asyncio.create_task(
            Evaluator(backend=backend(client, max_retries=2)).aevaluate([sample])
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.is_closed
    assert len(requests) == 1


@pytest.mark.parametrize("failure", [408, 409, 429, 500, 503, "timeout"])
async def test_transient_errors_retry_with_bounded_backoff(failure, monkeypatch, sample):
    calls, delays = [], []

    async def no_wait(delay):
        delays.append(delay)

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            if failure == "timeout":
                raise httpx2.ReadTimeout("private detail", request=request)
            return httpx2.Response(failure, text="private detail")
        return httpx2.Response(200, json=wire_response())

    monkeypatch.setattr(systemone.asyncio, "sleep", no_wait)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        result = await Evaluator(backend=backend(client, max_retries=2)).aevaluate_one(sample)
    assert len(calls) == 3
    assert len(delays) == 2
    assert 0 < delays[0] <= delays[1] <= 60
    assert result.passed is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_errors_are_not_retried_or_exposed(status, sample):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(status, text=f"private body {API_KEY} {BASE_URL}")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        configured = backend(client, api_key=API_KEY, max_retries=2)
        with pytest.raises(SystemOneError) as caught:
            await Evaluator(backend=configured).aevaluate_one(sample)
        assert not client.is_closed
    assert len(requests) == 1
    public_error = "\n".join(
        (str(caught.value), repr(caught.value), "".join(traceback.format_exception(caught.value)))
    )
    assert all(secret not in public_error for secret in (API_KEY, BASE_URL, "private body"))
    assert API_KEY not in repr(configured)


async def test_retry_exhaustion_records_unavailability_without_transport_details(
    monkeypatch, sample
):
    calls = []

    async def no_wait(delay):
        pass

    def handler(request):
        calls.append(request)
        raise httpx2.ConnectError(f"private detail {API_KEY} {BASE_URL}", request=request)

    monkeypatch.setattr(systemone.asyncio, "sleep", no_wait)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        result = await Evaluator(
            backend=backend(client, max_retries=2), errors="record"
        ).aevaluate_one(sample)
    assert len(calls) == 3
    assert result.passed is None
    assert result.metrics["answer_relevancy"].status == "error"
    exported = result.model_dump_json()
    assert all(secret not in exported for secret in (API_KEY, BASE_URL, "private detail"))


@pytest.mark.parametrize(
    "answer",
    [
        None,
        [],
        "0.9",
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": "0.9"},
        {"type": "noul", "noul": 2},
    ],
)
async def test_invalid_answer_preserves_successful_sibling(answer, sample):
    metrics = [
        Metric(name=name, instructions="Pass?", pass_definition="Pass") for name in ("good", "bad")
    ]
    answers = {"good": {"type": "noul", "noul": 0.9}, "bad": answer}
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=wire_response(answers))
        )
    ) as client:
        result = await Evaluator(metrics, backend=backend(client), errors="record").aevaluate_one(
            sample
        )
    assert result.metrics["good"].status == "ok"
    assert result.metrics["good"].raw_score == 0.9
    assert result.metrics["bad"].status == "error"
    assert result.metrics["bad"].raw_score is result.metrics["bad"].passed is None
    assert result.passed is None


@pytest.mark.parametrize(
    "kind,answer",
    [
        (
            "classification",
            {
                "type": "choice",
                "choice": "good",
                "confidence": True,
                "probabilities": {"good": 0.8, "bad": 0.2},
            },
        ),
        (
            "classification",
            {
                "type": "choice",
                "choice": "good",
                "confidence": 0.8,
                "probabilities": {"good": "0.8", "bad": 0.2},
            },
        ),
        (
            "classification",
            {
                "type": "choice",
                "choice": "unknown",
                "confidence": 0.8,
                "probabilities": {"good": 0.8, "bad": 0.2},
            },
        ),
        (
            "ordinal",
            {
                "type": "score",
                "score": "1.5",
                "confidence": 0.75,
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            },
        ),
        (
            "ordinal",
            {
                "type": "score",
                "score": True,
                "confidence": 0.75,
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            },
        ),
        (
            "ordinal",
            {
                "type": "score",
                "score": 0.5,
                "confidence": 0.75,
                "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            },
        ),
    ],
)
async def test_choice_and_score_wire_values_are_never_coerced(kind, answer, sample):
    metric = next(metric for metric in panel() if metric.name == kind)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=wire_response({kind: answer}))
        )
    ) as client:
        async with backend(client).session() as session:
            response = await session.judge(
                {"input": sample.input, "response": sample.response}, {kind: metric.question()}
            )
        assert response.answers[kind] == answer
        with pytest.raises(InvalidAnswerError):
            metric.read_answer(response.answers[kind])
        result = await Evaluator([metric], backend=backend(client), errors="record").aevaluate_one(
            sample
        )
    assert result.metrics[kind].status == "error"
    assert result.passed is None


@pytest.mark.parametrize(
    "kind,changes",
    [
        ("classification", {"choice": []}),
        ("classification", {"choice": {}}),
        ("predicate", {"noul": 10**1000}),
        ("classification", {"confidence": 10**1000}),
        ("classification", {"probabilities": {"good": 10**1000, "bad": 0.2}}),
        ("ordinal", {"score": 10**1000}),
    ],
    ids=[
        "choice-list",
        "choice-object",
        "huge-noul",
        "huge-confidence",
        "huge-probability",
        "huge-score",
    ],
)
async def test_adversarial_raw_values_preserve_successful_sibling(kind, changes, sample):
    metric = next(metric for metric in panel() if metric.name == kind)
    good = Metric(name="good", instructions="Pass?", pass_definition="Pass")
    valid_answers = {
        "predicate": {"type": "noul", "noul": 0.9},
        "classification": {
            "type": "choice",
            "choice": "good",
            "confidence": 0.8,
            "probabilities": {"good": 0.8, "bad": 0.2},
        },
        "ordinal": {
            "type": "score",
            "score": 1.5,
            "confidence": 0.75,
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
        },
    }
    invalid = {**valid_answers[kind], **changes}
    answers = {"good": {"type": "noul", "noul": 0.9}, kind: invalid}
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=wire_response(answers))
        )
    ) as client:
        async with backend(client).session() as session:
            response = await session.judge(
                {"input": sample.input, "response": sample.response},
                {good.name: good.question(), kind: metric.question()},
            )
        assert response.answers[kind] == invalid
        with pytest.raises(InvalidAnswerError):
            metric.read_answer(response.answers[kind])
        result = await Evaluator(
            [good, metric], backend=backend(client), errors="record"
        ).aevaluate_one(sample)
    assert result.metrics["good"].status == "ok"
    assert result.metrics["good"].raw_score == 0.9
    assert result.metrics[kind].status == "error"
    assert result.metrics[kind].raw_score is result.metrics[kind].passed is None
    assert result.passed is None


@pytest.mark.parametrize("total", [0, 2])
async def test_large_choice_broken_distribution_is_unavailable_and_runtime_blocks(total, sample):
    criteria = {str(index): f"Option {index}" for index in range(200)}
    metric = Metric(
        name="many",
        kind="choice",
        instructions="Choose an option",
        criteria=criteria,
        pass_options=tuple(criteria)[:-1],
        pass_definition="One of the first 199 options",
    )
    good = Metric(name="good", instructions="Pass?", pass_definition="Pass")
    invalid = {
        "type": "choice",
        "choice": "0",
        "confidence": 0.9,
        "probabilities": {option: total / len(criteria) for option in criteria},
    }
    answers = {"good": {"type": "noul", "noul": 0.9}, "many": invalid}
    calls = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=wire_response(answers))
        )
    ) as client:
        evaluator = Evaluator([good, metric], backend=backend(client), errors="record")
        result = await evaluator.aevaluate_one(sample)
        assert result.metrics["good"].raw_score == 0.9
        assert result.metrics["many"].status == "error"
        assert result.metrics["many"].raw_score is result.metrics["many"].passed is None
        assert result.passed is None
        guard = RuntimeGuard({"step": GuardPolicy(evaluator)})
        with pytest.raises(GuardrailViolation) as caught:
            await guard.arun("step", sample, lambda: calls.append("executed"))
    assert not calls
    assert caught.value.decision.action == "block"
    assert caught.value.decision.unavailable_metrics == ("many",)
    assert caught.value.decision.evaluation.metrics["good"].raw_score == 0.9


async def test_missing_answer_is_independently_unavailable(sample):
    metrics = [
        Metric(name=name, instructions="Pass?", pass_definition="Pass")
        for name in ("good", "missing")
    ]
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, json=wire_response({"good": {"type": "noul", "noul": 0.9}})
            )
        )
    ) as client:
        async with backend(client).session() as session:
            result = await session.judge(
                {"input": sample.input, "response": sample.response},
                {metric.name: metric.question() for metric in metrics},
            )
    assert result.answers == {"good": {"type": "noul", "noul": 0.9}}
    assert result.answer_errors == {"missing": "invalid_answer"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model": MODEL},
        {"model": None, "answers": {}},
        {"model": True, "answers": {}},
        {"model": "", "answers": {}},
        {"model": MODEL, "answers": []},
        wire_response(
            {
                "answer_relevancy": {"type": "noul", "noul": 0.9},
                "unknown": {"type": "noul", "noul": 0.8},
            }
        ),
    ],
)
async def test_malformed_top_level_or_unknown_names_fail_whole_response(payload, sample):
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=payload))
    ) as client:
        with pytest.raises(InvalidAnswerError):
            await Evaluator(backend=backend(client)).aevaluate_one(sample)


@pytest.mark.parametrize(
    "content",
    [
        '{"model":"open-laya","model":"other","answers":{}}',
        '{"model":"open-laya","answers":{"answer_relevancy":{"type":"noul","noul":0.9,"noul":0.1}}}',
        '{"model":"open-laya","answers":{"answer_relevancy":{"type":"noul","noul":0.9},"answer_relevancy":{"type":"noul","noul":0.1}}}',
        "not json",
    ],
)
async def test_invalid_json_and_duplicate_fields_fail_closed(content, sample):
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, text=content))
    ) as client:
        with pytest.raises(InvalidAnswerError):
            await Evaluator(backend=backend(client)).aevaluate_one(sample)


async def test_usage_preserves_flat_counters_and_ignores_laya_diagnostics(sample):
    usage = {
        "input_tokens": 12,
        "output_tokens": 0,
        "total_tokens": 12,
        "cached_tokens": None,
        "truncated": True,
        "truncated_questions": ["answer_relevancy"],
        "details": {"private": "ignored"},
    }
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=wire_response(usage=usage))
        )
    ) as client:
        result = await Evaluator(backend=backend(client)).aevaluate_one(sample)
    assert result.usage == {
        "input_tokens": 12,
        "output_tokens": 0,
        "total_tokens": 12,
        "cached_tokens": None,
    }


@pytest.mark.parametrize("name", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [True, "12", -1])
async def test_known_token_counters_do_not_coerce_malformed_values(name, value, sample):
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=wire_response(usage={name: value}))
        )
    ) as client:
        with pytest.raises(InvalidAnswerError):
            await Evaluator(backend=backend(client)).aevaluate_one(sample)


async def test_omitted_usage_is_empty(sample):
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=wire_response()))
    ) as client:
        result = await Evaluator(backend=backend(client)).aevaluate_one(sample)
    assert result.usage == {}


async def test_image_evidence_is_rejected_before_network(sample):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=wire_response())

    image_metric = Metric(
        name="visible",
        instructions="Visible?",
        required_fields=("input", "response", "images"),
        pass_definition="Visible",
    )
    image_sample = sample.model_copy(
        update={"images": (ImageInput.from_bytes(b"image", mime_type="image/png"),)}
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(UnsupportedModalityError):
            await Evaluator([image_metric], backend=backend(client)).aevaluate_one(image_sample)
        with pytest.raises(UnsupportedModalityError):
            async with backend(client).session() as session:
                await session.judge(
                    image_sample.state({"input", "response", "images"}),
                    {image_metric.name: image_metric.question()},
                )
    assert not requests


def test_provenance_identifies_endpoint_and_order_excluding_transport_configuration():
    metrics = panel()
    questions = {metric.name: metric.question() for metric in metrics}
    original = backend_provenance(backend(), questions)
    configuration = original["configuration"]
    assert configuration["provider"] == configuration["backend"] == "systemone"
    assert configuration["compiler_version"] == "1"
    assert original == backend_provenance(
        SystemOneBackend(
            model=MODEL, base_url=f"{BASE_URL}/", api_key=API_KEY, timeout=2, max_retries=5
        ),
        questions,
    )
    assert API_KEY not in json.dumps(original)
    assert original != backend_provenance(
        SystemOneBackend(model=MODEL, base_url="https://other.example/prefix"), questions
    )
    assert original != backend_provenance(backend(), dict(reversed(list(questions.items()))))
    reordered = metrics[1].model_copy(update={"criteria": {"bad": "Bad", "good": "Good"}})
    changed_questions = {**questions, "classification": reordered.question()}
    assert original != backend_provenance(backend(), changed_questions)


def test_calibration_load_rejects_endpoint_drift_before_request(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    actual_client = httpx2.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        payload = json.loads(request.content)
        return httpx2.Response(
            200,
            json=wire_response(
                {
                    name: {"type": "noul", "noul": float(payload["state"]["response"])}
                    for name in payload["questions"]
                }
            ),
        )

    def factory(**kwargs):
        return actual_client(**kwargs, transport=httpx2.MockTransport(handler))

    monkeypatch.setattr(systemone.httpx2, "AsyncClient", factory)
    config = CalibrationConfig(
        enabled=True, algorithm="isotonic", min_samples=20, min_validation_samples=10
    )
    pipeline = EvaluationPipeline([AnswerRelevancy()], backend=backend(), calibration=config)
    pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    path = tmp_path / "systemone-calibration.json"
    pipeline.save_calibration(path)
    restored = EvaluationPipeline(backend=backend(api_key=API_KEY, timeout=7), calibration=config)
    restored.load_calibration(path)
    result = restored.evaluate_one(EvaluationSample(input="new", response="0.8"))
    assert result.metrics["answer_relevancy"].calibrated_probability == pytest.approx(0.6)
    calls_before = len(requests)
    changed = SystemOneBackend(model=MODEL, base_url="https://other.example/prefix")
    with pytest.raises(CalibrationMismatchError):
        EvaluationPipeline(backend=changed, calibration=config).load_calibration(path)
    assert len(requests) == calls_before


@pytest.mark.parametrize("errors", ["raise", "record"])
async def test_runtime_blocks_unavailable_answers(errors, sample):
    calls = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=wire_response({})))
    ) as client:
        guard = RuntimeGuard(
            {"step": GuardPolicy(Evaluator(backend=backend(client), errors=errors))}
        )
        with pytest.raises(GuardrailViolation) as caught:
            await guard.arun("step", sample, lambda: calls.append("executed"))
        assert not client.is_closed
    assert not calls
    assert caught.value.decision.action == "block"
