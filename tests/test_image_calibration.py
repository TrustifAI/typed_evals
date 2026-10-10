"""Calibrate image-aware metrics through the real SDK with mocked HTTP."""

import json

import pytest
from conftest import calibrated_rows

openai = pytest.importorskip("openai", minversion="3.26.0")
httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("sklearn")

from typed_evals import (  # noqa: E402
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    ImageInput,
    Metric,
    OpenAIDecisionsBackend,
)
from typed_evals.errors import CalibrationMismatchError, DataLeakageError  # noqa: E402

PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl9sAAAAASUVORK5CYII="
)
GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


def images():
    return (
        ImageInput(data_url=PNG, label="first"),
        ImageInput(data_url=GIF, detail="low", label="second"),
    )


def image_metric(*, threshold=0.7, fields=("input", "response", "images")):
    return Metric(
        name="image_consistency",
        instructions="Does response correctly describe the images for input?",
        pass_definition="The description matches the supplied images.",
        required_fields=fields,
        threshold=threshold,
    )


def rows(prefix, size):
    return [
        CalibrationExample(
            sample=row.sample.model_copy(update={"images": images()}),
            labels={"image_consistency": row.labels["answer_relevancy"]},
        )
        for row in calibrated_rows(prefix, size=size)
    ]


def config():
    return CalibrationConfig(
        enabled=True, algorithm="isotonic", min_samples=20, min_validation_samples=10
    )


@pytest.fixture
def image_api():
    class Service:
        def __init__(self):
            self.calls = []

        def handler(self, request):
            assert request.method == "POST"
            assert request.url.path == "/v1/decisions"
            body = json.loads(request.content)
            self.calls.append(body)
            assert body["input"][0]["role"] == "user"
            content = body["input"][0]["content"]
            assert content[0]["type"] == "input_text"
            state = json.loads(content[0]["text"])
            assert len(state["images"]) == len(content) - 1
            assert all(part["type"] == "input_image" for part in content[1:])
            return httpx2.Response(
                200,
                json={
                    "model": "gpt-6-luna-snapshot",
                    "answers": [
                        {
                            "name": "image_consistency",
                            "type": "predicate",
                            "probability": float(state["response"]),
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 0, "total_tokens": 12},
                },
            )

    return Service()


def client_for(api):
    return openai.AsyncOpenAI(
        api_key="test-only-not-real",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handler)),
    )


async def fit(backend):
    pipeline = EvaluationPipeline([image_metric()], backend=backend, calibration=config())
    await pipeline.afit(rows("image-train", 20), validation_data=rows("image-validation", 10))
    return pipeline


async def test_image_calibration_fit_save_load_apply_excludes_image_bytes(image_api, tmp_path):
    async with client_for(image_api) as client:
        backend = OpenAIDecisionsBackend(client=client)
        pipeline = await fit(backend)
        bundle = pipeline.calibration_bundle
        assert bundle.schema_version == 3
        assert bundle.evidence_fields == ("images", "input", "response")
        assert bundle.requested_model == "gpt-6-luna"
        assert bundle.observed_model == "gpt-6-luna-snapshot"
        assert bundle.curves["image_consistency"].x == pytest.approx((0.2, 0.8))
        assert bundle.curves["image_consistency"].y == pytest.approx((0.4, 0.6))
        assert len(image_api.calls) == 30
        configuration = bundle.backend_provenance["configuration"]
        assert configuration["compiler_version"] == "2"
        assert configuration["input_serialization"] == (
            "json-unicode-sorted-compact+ordered-image-parts-v1"
        )
        instructions = json.loads(configuration["questions"][0]["instructions"].split("\n")[0])
        assert instructions["required_modalities"] == ["text", "image"]

        artifact = tmp_path / "image-calibration.json"
        pipeline.save_calibration(artifact)
        exported = artifact.read_text()
        for secret in (PNG, GIF, "base64,", "test-only-not-real"):
            assert secret not in exported
        loaded = EvaluationPipeline(
            [image_metric(threshold=0.5)], backend=backend, calibration=config()
        ).load_calibration(artifact)
        result = await loaded.aevaluate_one(
            EvaluationSample(input="new image example", response="0.8", images=images())
        )
        judgment = result.metrics["image_consistency"]
        assert judgment.raw_score == pytest.approx(0.8)
        assert judgment.calibrated_probability == pytest.approx(0.6)
        assert judgment.calibration_target == "metric_pass"
        assert result.passed is True
        assert loaded.calibration_bundle.backend_provenance == bundle.backend_provenance


async def test_image_calibration_overlap_uses_source_order_detail_and_text(image_api):
    async with client_for(image_api) as client:
        pipeline = await fit(OpenAIDecisionsBackend(client=client))
        reserved = rows("image-train", 20)[0].sample
        calls = len(image_api.calls)
        # IDs and metadata do not make identical image and text evidence independent.
        duplicate = reserved.model_copy(update={"id": "different", "metadata": {"new": True}})
        with pytest.raises(DataLeakageError, match="reuses calibration"):
            await pipeline.aevaluate_one(duplicate)
        assert len(image_api.calls) == calls

        changed_source = reserved.images[0].model_copy(update={"data_url": GIF})
        changed_detail = reserved.images[0].model_copy(update={"detail": "high"})
        independent = [
            reserved.model_copy(update={"images": (changed_source, reserved.images[1])}),
            reserved.model_copy(update={"images": tuple(reversed(reserved.images))}),
            reserved.model_copy(update={"images": (changed_detail, reserved.images[1])}),
            reserved.model_copy(update={"input": "independent text with the same images"}),
        ]
        report = await pipeline.aevaluate(independent)
        assert len(image_api.calls) == calls + len(independent)
        assert all(
            result.metrics["image_consistency"].calibrated_probability == pytest.approx(0.4)
            for result in report.results
        )
        assert len({reserved.content_hash, *(sample.content_hash for sample in independent)}) == 5


@pytest.mark.parametrize("drift", ["selected_fields", "image_compiler"])
async def test_image_calibration_configuration_drift_requires_refit_before_requests(
    image_api, tmp_path, monkeypatch, drift
):
    async with client_for(image_api) as client:
        backend = OpenAIDecisionsBackend(client=client)
        pipeline = await fit(backend)
        artifact = tmp_path / "image-calibration.json"
        pipeline.save_calibration(artifact)
        metric = image_metric()
        if drift == "selected_fields":
            metric = image_metric(fields=("input", "response"))
            text_configuration = backend.calibration_provenance({metric.name: metric.question()})
            assert text_configuration["compiler_version"] == "1"
            assert text_configuration["input_serialization"] == "json-unicode-sorted-compact"
        else:
            monkeypatch.setattr(
                "typed_evals.backends.openai_decisions.IMAGE_COMPILER_VERSION", "changed"
            )
        calls = len(image_api.calls)
        other = EvaluationPipeline([metric], backend=backend, calibration=config())
        with pytest.raises(CalibrationMismatchError, match="refit"):
            other.load_calibration(artifact)
        assert len(image_api.calls) == calls


async def test_repeated_image_calibration_evidence_rejected_before_requests(image_api):
    async with client_for(image_api) as client:
        pipeline = EvaluationPipeline(
            [image_metric()], backend=OpenAIDecisionsBackend(client=client), calibration=config()
        )
        train = rows("image-train", 20)
        duplicate = CalibrationExample(
            sample=train[0].sample.model_copy(update={"id": "separate-record"}),
            labels=train[0].labels,
        )
        with pytest.raises(DataLeakageError, match="repeated sample content"):
            await pipeline.afit([*train, duplicate], validation_data=rows("image-validation", 10))
        assert not image_api.calls
