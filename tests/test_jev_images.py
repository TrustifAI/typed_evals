"""Exercise Jev image modes through the real SDK's multipart HTTP boundary."""

import json
from base64 import b64decode
from email import policy
from email.parser import BytesParser

import httpx2
import pytest
from conftest import calibrated_rows
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    ImageInput,
    JevBackend,
    Metric,
)
from typed_evals.backends.provenance import backend_provenance
from typed_evals.errors import CalibrationMismatchError

PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl9sAAAAASUVORK5CYII="
)
# The mocked service exercises byte transport, not JPEG decoding.
JPEG_BYTES = b"\xff\xd8\xff\xe0Jev JPEG transport fixture\xff\xd9"
IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"


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
        ImageInput(data_url=PNG, label="first"),
        ImageInput.from_bytes(JPEG_BYTES, mime_type="image/jpeg", label="second"),
    )


def image_sample(**updates):
    return EvaluationSample(
        input="Describe the images.",
        response="0.8",
        images=images(),
        metadata={"private": "not judge evidence"},
    ).model_copy(update=updates)


def multipart_request(request):
    """Read SDK-chosen references and part names without relying on their spelling."""
    assert request.headers["content-type"].startswith("multipart/form-data;")
    message = BytesParser(policy=policy.default).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: "
        + request.headers["content-type"].encode("ascii")
        + b"\r\n\r\n"
        + request.content
    )
    assert message.is_multipart()
    parts = {
        part.get_param("name", header="content-disposition"): part for part in message.iter_parts()
    }
    body = json.loads(parts.pop("request").get_payload(decode=True))
    return body, parts


def wire_response(body):
    return httpx2.Response(
        200,
        json={
            "model": body["model"],
            "answers": {
                name: {"type": "noul", "noul": float(body["state"]["response"])}
                for name in body["questions"]
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        },
    )


def sdk_client(handler):
    return AsyncTypeSafeClient(
        api_key="test-only-private-key",
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(handler),
    )


@pytest.mark.parametrize("mode", ["experimental", "native"])
async def test_jev_image_modes_send_ordered_native_assets_for_arbitrary_model_names(mode):
    calls = []
    sample = image_sample()
    metric = image_metric()

    def handler(request):
        calls.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/systemone"
        assert request.headers.get(IMAGE_HEADER) == ("true" if mode == "experimental" else None)
        body, parts = multipart_request(request)
        assert body["model"] == "future-vision-model"
        assert body["questions"] == {metric.name: metric.question().model_dump(mode="json")}
        assert set(body) == {"model", "state", "questions", "assets"}
        assert body["state"]["input"] == sample.input
        assert body["state"]["response"] == sample.response
        assert "metadata" not in body["state"]
        references = []
        for item, image, content_type in zip(
            body["state"]["images"], sample.images, ("image/png", "image/jpeg"), strict=True
        ):
            assert set(item) == {"image", "label"}
            assert item["label"] == image.label
            reference = item["image"]
            references.append(reference)
            asset = body["assets"][reference]
            assert asset["type"] == "image"
            assert set(asset) == {"type", "part"}
            part = parts[asset["part"]]
            assert part.get_content_type() == content_type
            assert part.get_payload(decode=True) == b64decode(image.data_url.partition(",")[2])
        assert len(set(references)) == len(parts) == len(body["assets"]) == 2
        assert "data_url" not in json.dumps(body)
        assert "base64," not in json.dumps(body)
        return wire_response(body)

    async with sdk_client(handler) as client:
        backend = JevBackend(model="future-vision-model", client=client, image_mode=mode)
        assert backend.supported_modalities == frozenset({"text", "image"})
        result = await Evaluator([metric], backend=backend).aevaluate_one(sample)
        assert result.metrics[metric.name].raw_score == pytest.approx(0.8)
        assert result.model == "future-vision-model"
        assert "data_url" not in result.model_dump_json()
        # The evaluator leaves a caller-owned SDK client usable.
        await Evaluator([metric], backend=backend).aevaluate_one(sample)
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["disabled", "experimental", "native"])
async def test_jev_text_requests_and_provenance_remain_unchanged_in_every_image_mode(mode):
    metric = AnswerRelevancy()
    sample = image_sample()
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers.get(IMAGE_HEADER) is None
        assert request.headers["content-type"].startswith("application/json")
        body = json.loads(request.content)
        assert body == {
            "model": "future-vision-model",
            "state": {"input": sample.input, "response": sample.response},
            "questions": {metric.name: metric.question().model_dump(mode="json")},
        }
        return wire_response(body)

    async with sdk_client(handler) as client:
        backend = JevBackend(model="future-vision-model", client=client, image_mode=mode)
        original = JevBackend(model=backend.model, client=client)
        questions = {metric.name: metric.question()}
        assert backend_provenance(backend, questions) == backend_provenance(original, questions)
        assert "image_mode" not in backend.calibration_provenance(questions)
        assert "input_serialization" not in backend.calibration_provenance(questions)
        await Evaluator([metric], backend=backend).aevaluate_one(sample)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["experimental", "native"])
@pytest.mark.parametrize(
    "image",
    [
        ImageInput.from_bytes(b"GIF89a fixture", mime_type="image/gif"),
        ImageInput.from_bytes(b"RIFF fixture WEBP", mime_type="image/webp"),
        ImageInput(data_url=PNG, detail="low"),
        ImageInput(data_url=PNG, detail="high"),
        ImageInput(data_url=PNG, detail="original"),
    ],
)
async def test_jev_image_limits_preflight_the_whole_batch_and_direct_sessions(mode, image):
    class PreflightBackend(JevBackend):
        def session(self):
            pytest.fail("Invalid image evidence must fail before opening the batch session")

    invalid = image_sample(images=(image,))
    backend = PreflightBackend(image_mode=mode)
    with pytest.raises(ValueError):
        await Evaluator([image_metric()], backend=backend).aevaluate([image_sample(), invalid])

    # Direct session use cannot bypass the provider's image restrictions.
    backend = JevBackend(client=object(), image_mode=mode)
    async with backend.session() as session:
        with pytest.raises(ValueError):
            await session.judge(invalid.state({"input", "response", "images"}), {})


@pytest.mark.parametrize("invalid", ["not a sequence", {"data_url": PNG}, 42])
async def test_jev_direct_sessions_reject_invalid_image_sequences(invalid):
    backend = JevBackend(client=object(), image_mode="native")
    async with backend.session() as session:
        with pytest.raises(ValueError):
            await session.judge({"input": "Describe", "images": invalid}, {})


@pytest.mark.parametrize("invalid", ["automatic", "enabled", None])
def test_jev_rejects_unknown_image_modes(invalid):
    with pytest.raises(ValueError, match="image_mode"):
        JevBackend(image_mode=invalid)


@pytest.mark.parametrize("mode", ["experimental", "native"])
def test_jev_image_provenance_records_compiler_and_transport_mode(mode):
    metric = image_metric()
    provenance = JevBackend(image_mode=mode).calibration_provenance(
        {metric.name: metric.question()}
    )
    assert provenance["compiler_version"] == "images-1"
    assert provenance["image_mode"] == mode
    assert provenance["input_serialization"] == "ordered-sdk-image-assets-v1"
    assert provenance["questions"][0]["schema"]["instructions"]["required_modalities"] == [
        "text",
        "image",
    ]


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


@pytest.mark.parametrize("mode", ["experimental", "native"])
async def test_jev_image_calibration_fit_save_load_apply_excludes_image_data(mode, tmp_path):
    pytest.importorskip("sklearn")
    calls = []

    def handler(request):
        body, _ = multipart_request(request)
        calls.append(body)
        return wire_response(body)

    async with sdk_client(handler) as client:
        backend = JevBackend(model="future-vision-model", client=client, image_mode=mode)
        pipeline = EvaluationPipeline(
            [image_metric()], backend=backend, calibration=calibration_config()
        )
        await pipeline.afit(
            image_rows("image-train", 20), validation_data=image_rows("image-validation", 10)
        )
        bundle = pipeline.calibration_bundle
        assert bundle.evidence_fields == ("images", "input", "response")
        assert bundle.requested_model == bundle.observed_model == "future-vision-model"
        assert len(calls) == 30
        configuration = bundle.backend_provenance["configuration"]
        assert configuration["compiler_version"] == "images-1"
        assert configuration["image_mode"] == mode
        assert configuration["input_serialization"] == "ordered-sdk-image-assets-v1"

        artifact = tmp_path / "jev-image-calibration.json"
        pipeline.save_calibration(artifact)
        exported = artifact.read_text(encoding="utf-8")
        for secret in (
            PNG,
            *(image.data_url.partition(",")[2] for image in images()),
            "base64,",
            "test-only-private-key",
            "Jev JPEG transport fixture",
        ):
            assert secret not in exported
        restored = EvaluationPipeline(
            [image_metric(threshold=0.5)], backend=backend, calibration=calibration_config()
        ).load_calibration(artifact)
        result = await restored.aevaluate_one(image_sample(input="independent image example"))
        judgment = result.metrics["image_consistency"]
        assert judgment.raw_score == pytest.approx(0.8)
        assert judgment.calibrated_probability == pytest.approx(0.6)
        assert result.passed is True

        other_mode = "native" if mode == "experimental" else "experimental"
        other = EvaluationPipeline(
            [image_metric()],
            backend=JevBackend(model=backend.model, client=client, image_mode=other_mode),
            calibration=calibration_config(),
        )
        requests_before = len(calls)
        with pytest.raises(CalibrationMismatchError, match="configuration changed.*refit"):
            other.load_calibration(artifact)
        assert len(calls) == requests_before
