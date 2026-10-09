"""Fit and reuse calibration from actual SDK Decisions HTTP responses."""

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
    JevBackend,
    Metric,
    OpenAIDecisionsBackend,
)
from typed_evals.errors import (  # noqa: E402
    CalibrationMismatchError,
    DataLeakageError,
    DecisionRefusalError,
)


def metric_panel(threshold=0.7):
    return [
        Metric(
            name="predicate",
            instructions="Is positive?",
            pass_definition="Positive",
            threshold=threshold,
        ),
        Metric(
            name="choice",
            kind="choice",
            instructions="Classify",
            criteria={"pass": "Pass", "fail": "Fail"},
            pass_options=("pass",),
            pass_definition="Pass",
            threshold=threshold,
        ),
        Metric(
            name="score",
            kind="score",
            instructions="Rate",
            criteria=("Low", "Middle", "High"),
            pass_definition="High",
            threshold=threshold,
        ),
    ]


def rows(prefix):
    return [
        CalibrationExample(
            sample=row.sample,
            labels={
                name: row.labels["answer_relevancy"] for name in ("predicate", "choice", "score")
            },
        )
        for row in calibrated_rows(prefix)
    ]


def wire(p, *, refusal=False, model="gpt-6-luna-snapshot"):
    return {
        "model": model,
        "answers": [
            {"name": "predicate", "type": "predicate", "probability": p},
            {
                "name": "choice",
                "type": "choice",
                "choice": "pass" if p > 0.5 else "fail",
                "confidence": 1 - p,
                "probabilities": [
                    {"value": "pass", "probability": p},
                    {"value": "fail", "probability": 1 - p},
                ],
            },
            {"name": "score", "type": "refusal"}
            if refusal
            else {
                "name": "score",
                "type": "score",
                "score": p * 2,
                "confidence": 1 - p,
                "probabilities": [
                    {"label": str(i), "value": i, "probability": [1 - p, 0, p][i]} for i in range(3)
                ],
            },
        ],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 12,
        },
    }


@pytest.fixture
def api():
    class Service:
        refuse = False
        model = "gpt-6-luna-snapshot"

        def __init__(self):
            self.calls = []

        def handler(self, request):
            body = json.loads(request.content)
            self.calls.append(body)
            return httpx2.Response(
                200,
                json=wire(
                    float(json.loads(body["input"])["response"]),
                    refusal=self.refuse,
                    model=self.model,
                ),
            )

    return Service()


def config():
    return CalibrationConfig(enabled=True, min_samples=20, min_validation_samples=10)


def client_for(api):
    return openai.AsyncOpenAI(
        api_key="test-only-not-real",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handler)),
    )


async def fit(backend):
    pipeline = EvaluationPipeline(metric_panel(), backend=backend, calibration=config())
    await pipeline.afit(rows("train"), validation_data=rows("validation"))
    return pipeline


async def test_native_fit_save_load_raw_not_provider_confidence_and_threshold_compatibility(
    api, tmp_path
):
    async with client_for(api) as client:
        backend = OpenAIDecisionsBackend(client=client)
        pipeline = await fit(backend)
        bundle = pipeline.calibration_bundle
        assert bundle.schema_version == 2
        configuration = bundle.backend_provenance["configuration"]
        assert configuration["provider"] == "openai"
        assert configuration["backend"] == "openai-decisions"
        assert configuration["compiler_version"] == "1"
        assert [q["name"] for q in configuration["questions"]] == ["predicate", "choice", "score"]
        assert bundle.requested_model == "gpt-6-luna"
        assert bundle.observed_model == "gpt-6-luna-snapshot"
        for curve in bundle.curves.values():
            assert curve.x == pytest.approx((0.2, 0.8))
            assert curve.y == pytest.approx((0.4, 0.6))
        assert len(api.calls) == 80
        assert all(len(body["questions"]) == 3 for body in api.calls)
        artifact = tmp_path / "decisions-calibration.json"
        pipeline.save_calibration(artifact)
        exported = artifact.read_text()
        assert "test-only-not-real" not in exported
        assert '"timeout"' not in exported
        assert '"max_retries"' not in exported
        assert '"threshold"' not in json.dumps(configuration)
        reloaded = EvaluationPipeline(
            metric_panel(threshold=0.5),
            backend=OpenAIDecisionsBackend(client=client, timeout=72, max_retries=9),
            calibration=config(),
        ).load_calibration(artifact)
        result = await reloaded.aevaluate_one(EvaluationSample(input="new test", response="0.8"))
        assert result.passed is True
        for metric in result.metrics.values():
            assert metric.raw_score == pytest.approx(0.8)
            assert metric.calibrated_probability == pytest.approx(0.6)
            assert metric.calibration_target == "metric_pass"
        assert result.metrics["choice"].confidence == pytest.approx(0.2)
        assert result.metrics["score"].confidence == pytest.approx(0.2)
        assert result.metrics["predicate"].confidence is None
        with pytest.raises(DataLeakageError):
            await reloaded.aevaluate([rows("train")[0].sample])


@pytest.mark.parametrize(
    "drift", ["provider", "compiler", "requested_model", "observed_model", "legacy", "endpoint"]
)
async def test_native_calibration_rejects_incompatible_configuration_before_requests(
    api, tmp_path, monkeypatch, drift
):
    async with client_for(api) as client:
        pipeline = await fit(OpenAIDecisionsBackend(client=client))
        artifact = tmp_path / "calibration.json"
        pipeline.save_calibration(artifact)
        backend = OpenAIDecisionsBackend(client=client)
        if drift == "provider":
            backend = JevBackend(model="gpt-6-luna")
        elif drift == "compiler":
            monkeypatch.setattr("typed_evals.backends.openai_decisions.COMPILER_VERSION", "changed")
        elif drift == "requested_model":
            backend = OpenAIDecisionsBackend(model="different", client=client)
        elif drift == "endpoint":
            client.base_url = "https://other-provider.example/v1"
        elif drift == "legacy":
            data = json.loads(artifact.read_text())
            data["schema_version"] = 1
            data.pop("backend_provenance")
            artifact.write_text(json.dumps(data))
        other = EvaluationPipeline(metric_panel(), backend=backend, calibration=config())
        calls = len(api.calls)
        if drift == "observed_model":
            other.load_calibration(artifact)
            api.model = "gpt-6-luna-changed"
            with pytest.raises(CalibrationMismatchError, match="model version"):
                await other.aevaluate_one(EvaluationSample(input="new", response="0.8"))
            assert len(api.calls) == calls + 1
        else:
            with pytest.raises(CalibrationMismatchError, match="refit"):
                other.load_calibration(artifact)
            assert len(api.calls) == calls


async def test_native_refused_refit_preserves_previous_valid_bundle(api):
    async with client_for(api) as client:
        pipeline = await fit(OpenAIDecisionsBackend(client=client))
        previous = pipeline.calibration_bundle
        api.refuse = True
        with pytest.raises(DecisionRefusalError):
            await pipeline.afit(rows("refit"), validation_data=rows("refit-validation"))
        assert pipeline.calibration_bundle is previous
        api.refuse = False
        result = await pipeline.aevaluate_one(EvaluationSample(input="new", response="0.8"))
        assert result.metrics["score"].calibrated_probability == pytest.approx(0.6)


async def test_endpoint_identity_normalization_and_credentials_excluded(monkeypatch):
    questions = {metric.name: metric.question() for metric in metric_panel()}
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    default = OpenAIDecisionsBackend().calibration_provenance(questions)
    async with openai.AsyncOpenAI(
        api_key="secret-key", base_url="https://api.openai.com:443/v1/"
    ) as client:
        injected = OpenAIDecisionsBackend(
            client=client, timeout=70, max_retries=17
        ).calibration_provenance(questions)
        assert injected == default
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://username:secret-password@api.openai.com:443/v1/?api_key=secret-query#secret-fragment",
    )
    assert OpenAIDecisionsBackend(api_key="secret-key").calibration_provenance(questions) == default
    assert "secret" not in json.dumps(default)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    assert OpenAIDecisionsBackend().calibration_provenance(questions) != default
