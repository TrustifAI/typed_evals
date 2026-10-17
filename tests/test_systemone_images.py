"""Opt-in System One image transport through the shared backend evidence API."""

import copy
import json

import httpx2
import pytest
from conftest import calibrated_rows

from typed_evals import (
    AnswerRelevancy,
    BaseBackend,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    ImageInput,
    Metric,
    SystemOneBackend,
)
from typed_evals.backends import systemone
from typed_evals.backends.provenance import backend_provenance
from typed_evals.errors import CalibrationMismatchError, UnsupportedModalityError

MODEL = "future-vision-decision-model"
BASE_URL = "https://judge.example/prefix"
API_KEY = "private-image-test-key"


def image_metric(*, threshold=0.7):
    return Metric(
        name="image_consistency",
        instructions="Does the response correctly describe the attached images?",
        pass_definition="The description matches the attached images.",
        required_fields=("input", "response", "images"),
        threshold=threshold,
    )


def images():
    return (
        ImageInput.from_bytes(
            b"first image fixture", mime_type="image/png", label="商品", detail="high"
        ),
        ImageInput.from_bytes(b"second image fixture", mime_type="image/jpeg", detail="low"),
        ImageInput.from_bytes(
            b"third image fixture", mime_type="image/webp", label="third", detail="auto"
        ),
    )


def image_sample(**updates):
    return EvaluationSample(
        input="Describe the images.",
        response="0.8",
        images=images(),
        metadata={"private": "not judge evidence"},
    ).model_copy(update=updates)


def backend(client=None, **kwargs):
    return SystemOneBackend(
        model=MODEL,
        base_url=BASE_URL,
        client=client,
        max_retries=kwargs.pop("max_retries", 0),
        image_mode=kwargs.pop("image_mode", "native"),
        **kwargs,
    )


def wire_response(body, answers=None):
    return httpx2.Response(
        200,
        json={
            "model": body["model"],
            "answers": answers
            if answers is not None
            else {
                name: {"type": "noul", "noul": float(body["state"]["response"])}
                for name in body["questions"]
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        },
    )


async def test_native_images_and_mixed_primitives_share_one_filtered_request():
    metrics = [
        image_metric(),
        Metric(
            name="category",
            kind="choice",
            instructions="Classify",
            criteria={"good": "Good", "bad": "Bad"},
            pass_options=("good",),
            pass_definition="Good",
        ),
        Metric(
            name="quality",
            kind="score",
            instructions="Rate",
            criteria=("Poor", "Fair", "Good"),
            pass_definition="Good",
        ),
    ]
    answers = {
        "image_consistency": {"type": "noul", "noul": 0.8},
        "category": {
            "type": "choice",
            "choice": "good",
            "confidence": 0.8,
            "probabilities": {"good": 0.8, "bad": 0.2},
        },
        "quality": {
            "type": "score",
            "score": 1.5,
            "confidence": 0.75,
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
        },
    }
    requests = []

    def handler(request):
        requests.append(request)
        return wire_response(json.loads(request.content), answers)

    sample = image_sample(contexts=("unselected context",))
    snapshot = sample.model_dump()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.warns(UserWarning, match="contexts were supplied"):
            result = await Evaluator(
                metrics, backend=backend(client, api_key=API_KEY)
            ).aevaluate_one(sample)
        assert not client.is_closed
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/prefix/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers.get("TypeSafe-Danger-Zone-Images") is None
    assert request.headers["content-type"].startswith("application/json")
    body = json.loads(request.content)
    assert body == {
        "model": MODEL,
        "state": {
            "input": sample.input,
            "response": sample.response,
            "images": [
                {"index": i, "label": image.label, "detail": image.detail}
                for i, image in enumerate(sample.images)
            ],
        },
        "questions": {metric.name: metric.question().model_dump(mode="json") for metric in metrics},
        "images": [{"url": image.data_url, "detail": image.detail} for image in sample.images],
    }
    assert all(image.data_url not in json.dumps(body["state"]) for image in sample.images)
    assert "unselected context" not in request.content.decode()
    assert "not judge evidence" not in request.content.decode()
    assert sample.model_dump() == snapshot
    assert result.model == MODEL
    assert result.metrics["image_consistency"].raw_score == pytest.approx(0.8)
    assert result.metrics["category"].selected_choice == "good"
    assert result.metrics["quality"].raw_score == pytest.approx(0.75)
    assert result.usage == {"input_tokens": 12, "output_tokens": 0}
    assert all(image.data_url not in result.model_dump_json() for image in sample.images)


@pytest.mark.parametrize("mode", ["disabled", "native"])
async def test_unused_images_preserve_text_wire_contract_and_provenance(mode):
    metric = AnswerRelevancy()
    sample = image_sample(
        images=(ImageInput.from_bytes(b"ignored image", mime_type="image/gif", detail="original"),)
    )
    requests = []

    def handler(request):
        requests.append(request)
        return wire_response(json.loads(request.content))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        configured = backend(client, image_mode=mode)
        questions = {metric.name: metric.question()}
        original = backend(client, image_mode="disabled")
        assert backend_provenance(configured, questions) == backend_provenance(original, questions)
        provenance = configured.calibration_provenance(questions)
        assert provenance["compiler_version"] == "1"
        assert "image_mode" not in provenance
        assert "input_serialization" not in provenance
        await Evaluator([metric], backend=configured).aevaluate_one(sample)
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {
        "model": MODEL,
        "state": {"input": sample.input, "response": sample.response},
        "questions": {metric.name: metric.question().model_dump(mode="json")},
    }
    assert requests[0].headers.get("TypeSafe-Danger-Zone-Images") is None


@pytest.mark.parametrize("typed", [False, True])
async def test_direct_session_accepts_typed_and_json_image_evidence_without_mutation(typed):
    requests = []

    def handler(request):
        requests.append(request)
        return wire_response(json.loads(request.content))

    sample = image_sample()
    state = sample.state({"input", "response", "images"})
    if typed:
        state["images"] = sample.images
    snapshot = copy.deepcopy(state)
    metric = image_metric()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        async with backend(client).session() as session:
            response = await session.judge(state, {metric.name: metric.question()})
        assert not client.is_closed
    assert state == snapshot
    assert response.answers[metric.name]["noul"] == pytest.approx(0.8)
    assert json.loads(requests[0].content)["images"] == [
        {"url": image.data_url, "detail": image.detail} for image in sample.images
    ]


@pytest.mark.parametrize("errors", ["raise", "record"])
async def test_disabled_images_preflight_the_entire_batch_before_any_session(errors):
    class PreflightBackend(SystemOneBackend):
        def session(self):
            pytest.fail("Disabled image evidence must fail before opening a batch session")

    evaluator = Evaluator(
        [AnswerRelevancy(), image_metric()],
        backend=PreflightBackend(model=MODEL, base_url=BASE_URL),
        missing="skip",
        errors=errors,
    )
    with pytest.raises(UnsupportedModalityError):
        await evaluator.aevaluate(
            [EvaluationSample(input="text first", response="0.8"), image_sample()]
        )


async def test_original_detail_preflights_the_entire_native_batch():
    class PreflightBackend(SystemOneBackend):
        def session(self):
            pytest.fail("Unsupported image detail must fail before opening a batch session")

    invalid = image_sample(images=(images()[0].model_copy(update={"detail": "original"}),))
    with pytest.raises(ValueError, match="detail"):
        await Evaluator(
            [image_metric()],
            backend=PreflightBackend(model=MODEL, base_url=BASE_URL, image_mode="native"),
        ).aevaluate([image_sample(), invalid])


@pytest.mark.parametrize(
    "invalid", ["not a sequence", {"data_url": "bad"}, 42, [{"data_url": "bad"}]]
)
async def test_direct_sessions_reject_malformed_images_before_network(invalid):
    def handler(request):
        pytest.fail("Malformed image evidence must not reach the server")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        async with backend(client).session() as session:
            with pytest.raises(ValueError):
                await session.judge({"response": "0.8", "images": invalid}, {})


async def test_direct_sessions_cannot_bypass_image_mode_or_detail_limits():
    def handler(request):
        pytest.fail("Unsupported image evidence must not reach the server")

    metric = image_metric()
    questions = {metric.name: metric.question()}
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        async with backend(client, image_mode="disabled").session() as session:
            with pytest.raises(UnsupportedModalityError):
                await session.judge(image_sample().state({"images"}), questions)
        original = images()[0].model_copy(update={"detail": "original"})
        async with backend(client).session() as session:
            with pytest.raises(ValueError, match="detail"):
                await session.judge({"images": [original]}, questions)


@pytest.mark.parametrize("invalid", ["experimental", "enabled", None])
def test_unknown_image_modes_are_rejected(invalid):
    with pytest.raises(ValueError, match="image_mode"):
        backend(image_mode=invalid)


@pytest.mark.parametrize("state", [{}, {"images": None}, {"images": []}])
def test_shared_base_image_parser_accepts_missing_or_empty_images(state):
    assert BaseBackend.image_inputs(state) == ()


def test_shared_base_image_parser_accepts_mixed_typed_and_json_images():
    first, second, _ = images()
    state = {"images": [first, second.model_dump(mode="json")]}
    snapshot = copy.deepcopy(state)
    assert BaseBackend.image_inputs(state) == (first, second)
    assert state == snapshot


def test_shared_base_image_parser_revalidates_tampered_typed_images():
    malformed = images()[0].model_copy(update={"data_url": "https://invalid/image.png"})
    with pytest.raises(ValueError):
        BaseBackend.image_inputs({"images": [malformed]})
    with pytest.raises(ValueError):
        backend().validate_state({"images": [malformed]})


async def test_image_only_metric_sends_only_selected_image_evidence():
    metric = image_metric().model_copy(update={"required_fields": ("images",)})
    requests = []

    def handler(request):
        requests.append(request)
        return wire_response(
            json.loads(request.content), {metric.name: {"type": "noul", "noul": 0.8}}
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        result = await Evaluator([metric], backend=backend(client)).aevaluate_one(image_sample())
    assert result.passed
    body = json.loads(requests[0].content)
    assert set(body["state"]) == {"images"}
    assert len(body["images"]) == len(images())


async def test_transient_retries_preserve_image_body_and_caller_owned_client(monkeypatch):
    requests, delays = [], []

    async def no_wait(delay):
        delays.append(delay)

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx2.Response(503, text="retry")
        return wire_response(json.loads(request.content))

    monkeypatch.setattr(systemone.asyncio, "sleep", no_wait)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        configured = backend(client, max_retries=1)
        result = await Evaluator([image_metric()], backend=configured).aevaluate_one(image_sample())
        assert not client.is_closed
        assert (
            await Evaluator([image_metric()], backend=configured).aevaluate_one(image_sample())
        ).passed
    assert result.passed
    assert len(requests) == 3
    assert requests[0].content == requests[1].content == requests[2].content
    assert len(delays) == 1


def calibration_config():
    return CalibrationConfig(
        enabled=True, algorithm="isotonic", min_samples=20, min_validation_samples=10
    )


def image_rows(prefix, size):
    return [
        CalibrationExample(
            sample=row.sample.model_copy(update={"images": images()}),
            labels={"image_consistency": row.labels["answer_relevancy"]},
        )
        for row in calibrated_rows(prefix, size=size)
    ]


async def test_image_calibration_roundtrip_excludes_bytes_and_rejects_mode_or_compiler_drift(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    requests = []

    def handler(request):
        requests.append(request)
        return wire_response(json.loads(request.content))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        configured = backend(client, api_key=API_KEY)
        pipeline = EvaluationPipeline(
            [image_metric()], backend=configured, calibration=calibration_config()
        )
        await pipeline.afit(
            image_rows("image-train", 20), validation_data=image_rows("image-validation", 10)
        )
        bundle = pipeline.calibration_bundle
        assert bundle.evidence_fields == ("images", "input", "response")
        assert bundle.requested_model == bundle.observed_model == MODEL
        assert len(requests) == 30
        configuration = bundle.backend_provenance["configuration"]
        assert configuration["compiler_version"] == "images-1"
        assert configuration["image_mode"] == "native"
        assert configuration["input_serialization"] == "json-state+ordered-image-url-parts-v1"
        assert configuration["questions"][0]["schema"]["instructions"]["required_modalities"] == [
            "text",
            "image",
        ]

        artifact = tmp_path / "systemone-image-calibration.json"
        pipeline.save_calibration(artifact)
        exported = artifact.read_text(encoding="utf-8")
        for secret in (
            API_KEY,
            "base64,",
            *(image.data_url for image in images()),
            *(image.data_url.partition(",")[2] for image in images()),
            "first image fixture",
        ):
            assert secret not in exported
        restored = EvaluationPipeline(
            [image_metric(threshold=0.5)], backend=configured, calibration=calibration_config()
        ).load_calibration(artifact)
        result = await restored.aevaluate_one(image_sample(input="independent image example"))
        judgment = result.metrics["image_consistency"]
        assert judgment.raw_score == pytest.approx(0.8)
        assert judgment.calibrated_probability == pytest.approx(0.6)
        assert result.passed

        requests_before = len(requests)
        disabled = EvaluationPipeline(
            [image_metric()],
            backend=backend(client, image_mode="disabled"),
            calibration=calibration_config(),
        )
        with pytest.raises(CalibrationMismatchError, match="configuration changed.*refit"):
            disabled.load_calibration(artifact)
        monkeypatch.setattr(systemone, "IMAGE_COMPILER_VERSION", "future-images-2")
        with pytest.raises(CalibrationMismatchError, match="configuration changed.*refit"):
            EvaluationPipeline(
                [image_metric()], backend=configured, calibration=calibration_config()
            ).load_calibration(artifact)
        assert len(requests) == requests_before
