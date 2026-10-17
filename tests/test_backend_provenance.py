"""Calibration migration and optional backend identity remain provider independent."""

import json

import httpx2
import pytest
from conftest import FakeBackend, calibrated_rows
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
from typesafe_sdk.constants import DEFAULT_BASE_URL

from typed_evals import (
    AnswerRelevancy,
    CalibrationBundle,
    CalibrationConfig,
    EvaluationPipeline,
    EvaluationSample,
    JevBackend,
    Metric,
)
from typed_evals._utils import digest
from typed_evals.backends.provenance import backend_provenance
from typed_evals.errors import CalibrationError, CalibrationMismatchError


class ProvenanceBackend(FakeBackend):
    def __init__(self, *, provider="openai", compiler_version="1", setting="default"):
        super().__init__(
            lambda state, questions: {
                name: {"type": "noul", "noul": float(state["response"])} for name in questions
            }
        )
        self.provider = provider
        self.compiler_version = compiler_version
        self.setting = setting

    def calibration_provenance(self, questions):
        return {
            "provider": self.provider,
            "backend": "openai-decisions" if self.provider == "openai" else "other",
            "compiler_version": self.compiler_version,
            "setting": self.setting,
            "questions": [
                {"name": name, "question": question.model_dump(mode="json")}
                for name, question in questions.items()
            ],
        }


def config():
    return CalibrationConfig(
        enabled=True, algorithm="isotonic", min_samples=20, min_validation_samples=10
    )


def fitted(backend):
    pipeline = EvaluationPipeline([AnswerRelevancy()], backend=backend, calibration=config())
    pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    return pipeline


def legacy_artifact_data(bundle, *, version=1):
    data = bundle.model_dump(mode="json")
    data["schema_version"] = version
    if version == 1:
        del data["backend_provenance"]
    for curve in data["curves"].values():
        del curve["algorithm"]
    for metric in data["report"]["metrics"].values():
        del metric["algorithm"]
        del metric["log_loss_improved"]
    return data


def legacy_artifact(bundle, path, *, version=1):
    data = legacy_artifact_data(bundle, version=version)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_new_custom_fit_explicitly_records_unknown_identity_and_remains_usable(
    scoring_backend, tmp_path
):
    pipeline = fitted(scoring_backend)
    assert pipeline.calibration_bundle.schema_version == 3
    assert pipeline.calibration_bundle.backend_provenance == {"status": "unknown"}
    path = tmp_path / "custom.json"
    pipeline.save_calibration(path)
    restored = EvaluationPipeline(backend=scoring_backend, calibration=config())
    restored.load_calibration(path)
    result = restored.evaluate_one(EvaluationSample(input="new", response="0.8"))
    assert result.metrics["answer_relevancy"].calibrated_probability == pytest.approx(0.6)


@pytest.mark.parametrize(
    "changed",
    [
        {"provider": "different"},
        {"compiler_version": "2"},
        {"setting": "different"},
    ],
)
def test_static_provider_compiler_and_score_configuration_mismatch_before_request(
    tmp_path, changed
):
    pipeline = fitted(ProvenanceBackend())
    path = tmp_path / "known.json"
    pipeline.save_calibration(path)
    backend = ProvenanceBackend(**changed)
    restored = EvaluationPipeline(backend=backend, calibration=config())
    with pytest.raises(CalibrationMismatchError, match="backend, compiler.*refit"):
        restored.load_calibration(path)
    assert backend.sessions == 0


def test_evaluation_revalidates_backend_after_fit():
    backend = ProvenanceBackend()
    pipeline = fitted(backend)
    sessions_before = backend.sessions
    backend.compiler_version = "2"
    with pytest.raises(CalibrationMismatchError, match="compiler"):
        pipeline.evaluate_one(EvaluationSample(input="new", response="0.8"))
    assert backend.sessions == sessions_before


def test_verified_artifact_requires_backend_for_public_validation():
    pipeline = fitted(ProvenanceBackend())
    with pytest.raises(CalibrationMismatchError, match="current backend provenance"):
        pipeline.calibration_bundle.validate_for(
            pipeline.evaluator.metrics, pipeline.evaluator.backend.model
        )


def test_threshold_changes_preserve_verified_calibration(tmp_path):
    pipeline = fitted(ProvenanceBackend())
    path = tmp_path / "known.json"
    pipeline.save_calibration(path)
    restored = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.9)], backend=ProvenanceBackend(), calibration=config()
    )
    restored.load_calibration(path)
    result = restored.evaluate_one(EvaluationSample(input="new", response="0.8"))
    assert result.metrics["answer_relevancy"].score == pytest.approx(0.6)
    assert result.passed is False


@pytest.mark.parametrize("source_known", [False, True])
def test_v2_unknown_and_verified_identities_cannot_be_interchanged(
    source_known, scoring_backend, tmp_path
):
    source = ProvenanceBackend() if source_known else scoring_backend
    destination = scoring_backend if source_known else ProvenanceBackend()
    pipeline = fitted(source)
    path = tmp_path / "identity.json"
    legacy_artifact(pipeline.calibration_bundle, path, version=2)
    with pytest.raises(CalibrationMismatchError, match="refit"):
        EvaluationPipeline(backend=destination, calibration=config()).load_calibration(path)


def test_v1_custom_workflows_load_without_fabricating_identity(scoring_backend, tmp_path):
    pipeline = fitted(scoring_backend)
    path = tmp_path / "legacy.json"
    legacy_artifact(pipeline.calibration_bundle, path)
    bundle = CalibrationBundle.load(path)
    assert bundle.schema_version == 1
    assert bundle.backend_provenance is None
    restored = EvaluationPipeline(backend=scoring_backend, calibration=config())
    restored.load_calibration(path)
    assert restored.evaluate_one(EvaluationSample(input="new", response="0.8")).passed
    with pytest.raises(CalibrationMismatchError, match="Legacy.*provenance.*refit"):
        EvaluationPipeline(
            backend=JevBackend(model=scoring_backend.model), calibration=config()
        ).load_calibration(path)
    # Rejection never manufactures today's endpoint in the historical artifact.
    assert CalibrationBundle.load(path).backend_provenance is None


def test_v1_cannot_be_assumed_verified_for_decisions(scoring_backend, tmp_path):
    pipeline = fitted(scoring_backend)
    path = tmp_path / "legacy.json"
    legacy_artifact(pipeline.calibration_bundle, path)
    backend = ProvenanceBackend()
    with pytest.raises(CalibrationMismatchError, match="Legacy.*provenance.*refit"):
        EvaluationPipeline(backend=backend, calibration=config()).load_calibration(path)
    assert backend.sessions == 0


@pytest.mark.parametrize("version", [1, 2])
def test_artifact_schema_cannot_silently_add_or_drop_provenance(version, tmp_path):
    bundle = fitted(ProvenanceBackend()).calibration_bundle
    data = legacy_artifact_data(bundle, version=version)
    if version == 2:
        del data["backend_provenance"]
    else:
        data["backend_provenance"] = bundle.backend_provenance
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationError, match="artifact"):
        CalibrationBundle.load(path)


def test_corrupt_provenance_fingerprint_is_rejected(tmp_path):
    data = fitted(ProvenanceBackend()).calibration_bundle.model_dump(mode="json")
    data["backend_provenance"]["configuration"]["compiler_version"] = "2"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CalibrationError, match="artifact"):
        CalibrationBundle.load(path)


async def test_jev_provenance_excludes_transport_credentials_and_thresholds(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    first = AnswerRelevancy(threshold=0.5)
    changed = AnswerRelevancy(threshold=0.9)
    original = backend_provenance(JevBackend(), {first.name: first.question()})
    async with AsyncTypeSafeClient(api_key="secret") as client:
        # Supported SDK clients expose the resolved URL through _config, not base_url.
        assert not hasattr(client, "base_url")
        other = backend_provenance(
            JevBackend(api_key="secret", timeout=5.0, max_retries=0, client=client),
            {changed.name: changed.question()},
        )
    assert original == other
    assert "secret" not in json.dumps(original)


def test_jev_provenance_fingerprints_emitted_choice_order():
    first = Metric(
        name="ordered",
        kind="choice",
        instructions="Choose an option",
        criteria={"a": "First", "b": "Second"},
        pass_options=("a",),
        pass_definition="First",
    )
    reversed_options = first.model_copy(update={"criteria": {"b": "Second", "a": "First"}})
    assert first.fingerprint == reversed_options.fingerprint
    original = backend_provenance(JevBackend(), {first.name: first.question()})
    reordered = backend_provenance(JevBackend(), {first.name: reversed_options.question()})
    assert original != reordered


def test_invalid_optional_capability_fails_before_judging():
    backend = ProvenanceBackend()
    backend.compiler_version = None
    with pytest.raises(CalibrationError, match="provenance"):
        fitted(backend)
    assert backend.sessions == 0


def test_optional_capability_configuration_is_snapshotted():
    backend = ProvenanceBackend()
    configuration = {
        "provider": "custom",
        "backend": "custom",
        "compiler_version": "1",
        "settings": {"signal": "original"},
    }
    backend.calibration_provenance = lambda questions: configuration
    provenance = backend_provenance(backend, {})
    configuration["settings"]["signal"] = "changed"
    assert provenance["configuration"]["settings"]["signal"] == "original"


def jev_client(endpoint=DEFAULT_BASE_URL, *, requests=None, api_key="test-only-secret"):
    def handler(request):
        if requests is not None:
            requests.append(request)
        body = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "model": body["model"],
                "answers": {
                    name: {"type": "noul", "noul": float(body["state"]["response"])}
                    for name in body["questions"]
                },
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        )

    return AsyncTypeSafeClient(
        api_key=api_key,
        base_url=endpoint,
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(handler),
    )


async def fitted_jev(client):
    pipeline = EvaluationPipeline(
        [AnswerRelevancy()], backend=JevBackend(client=client), calibration=config()
    )
    await pipeline.afit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    return pipeline


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("https://API.TypeSafe.AI:443///", "https://api.typesafe.ai/"),
        (
            "https://user:URL_PRIVATE@judge.example:8443/gateway/v2/?key=QUERY_PRIVATE#FRAGMENT_PRIVATE",
            "https://judge.example:8443/gateway/v2/",
        ),
        ("http://Judge.Example:80/service", "http://judge.example/service/"),
        ("https://[2001:db8::1]:8443/service/", "https://[2001:db8::1]:8443/service/"),
    ],
)
async def test_jev_endpoint_normalization_preserves_routing_and_excludes_credentials(
    endpoint, expected
):
    async with jev_client(endpoint) as client:
        owned = JevBackend(base_url=endpoint).calibration_provenance({})
        injected = JevBackend(client=client).calibration_provenance({})
    assert owned == injected
    assert owned["endpoint"] == expected
    assert "PRIVATE" not in json.dumps(owned)
    assert owned["compiler_version"] == "2"


async def test_jev_default_explicit_and_injected_routes_have_identical_provenance(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    default = JevBackend().calibration_provenance({})
    explicit = JevBackend(base_url="https://API.TypeSafe.AI:443/").calibration_provenance({})
    async with jev_client() as client:
        injected = JevBackend(client=client).calibration_provenance({})
    assert default == explicit == injected
    assert default["endpoint"] == DEFAULT_BASE_URL + "/"


@pytest.mark.parametrize(
    "explicit,environment",
    [
        (None, None),
        (None, "   "),
        (None, "  https://env.example:8443/gateway/  "),
        ("https://explicit.example/custom", "https://ignored.example"),
    ],
)
async def test_jev_owned_endpoint_resolution_matches_supported_sdk(
    monkeypatch, explicit, environment
):
    if environment is None:
        monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_BASE_URL", environment)
    async with AsyncTypeSafeClient(api_key="test-only-secret", base_url=explicit) as client:
        assert JevBackend(base_url=explicit).calibration_provenance({}) == JevBackend(
            client=client
        ).calibration_provenance({})


async def test_jev_environment_changes_owned_routing_but_not_injected_client(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://original.example/gateway/")
    async with AsyncTypeSafeClient(api_key="test-only-secret") as client:
        owned = JevBackend()
        injected = JevBackend(client=client, base_url="https://ignored.example/constructor")
        original = injected.calibration_provenance({})
        assert owned.calibration_provenance({}) == original
        monkeypatch.setenv("TYPESAFE_BASE_URL", "https://changed.example/gateway/")
        assert owned.calibration_provenance({}) != original
        assert injected.calibration_provenance({}) == original
        assert (
            JevBackend(base_url="https://original.example/gateway").calibration_provenance({})
            == original
        )


@pytest.mark.parametrize("endpoint", ["", "judge.example", "file:///local", "https://bad:port"])
def test_jev_invalid_endpoint_fails_without_exposing_url(endpoint):
    with pytest.raises(CalibrationError, match="HTTP\\(S\\).*API root") as error:
        JevBackend(base_url=endpoint).calibration_provenance({})
    assert endpoint not in str(error.value) or not endpoint


def test_jev_injected_client_without_resolved_endpoint_fails_clearly(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", DEFAULT_BASE_URL)
    with pytest.raises(CalibrationError, match="injected SDK client's endpoint.*_config.base_url"):
        JevBackend(client=object(), base_url=DEFAULT_BASE_URL).calibration_provenance({})


@pytest.mark.parametrize(
    "destination",
    [
        "https://different.example/root",
        "https://original.example:8443/root",
        "https://original.example/other",
    ],
)
async def test_jev_endpoint_mismatch_rejects_artifact_before_http(tmp_path, destination):
    requests = []
    async with jev_client("https://original.example/root", requests=requests) as source:
        pipeline = await fitted_jev(source)
        artifact = tmp_path / "jev.json"
        pipeline.save_calibration(artifact)
        calls = len(requests)
        async with jev_client(destination, requests=requests) as client:
            restored = EvaluationPipeline(backend=JevBackend(client=client), calibration=config())
            with pytest.raises(CalibrationMismatchError, match="configuration changed.*refit"):
                restored.load_calibration(artifact)
        assert len(requests) == calls
        assert pipeline.calibration_bundle.requested_model == JevBackend().model
        assert pipeline.calibration_bundle.observed_model == JevBackend().model


async def test_jev_saved_artifact_excludes_keys_and_accepts_threshold_changes(tmp_path):
    api_key = "API_PRIVATE_SENTINEL"
    endpoint = "https://user:URL_PRIVATE_SENTINEL@judge.example/root?key=QUERY_PRIVATE_SENTINEL#FRAGMENT_PRIVATE_SENTINEL"
    async with jev_client(endpoint, api_key=api_key) as source:
        pipeline = await fitted_jev(source)
        artifact = tmp_path / "jev.json"
        pipeline.save_calibration(artifact)
    exported = artifact.read_text(encoding="utf-8")
    assert "PRIVATE_SENTINEL" not in exported
    assert '"timeout"' not in exported
    assert '"max_retries"' not in exported
    async with jev_client("https://judge.example/root/") as client:
        restored = EvaluationPipeline(
            [AnswerRelevancy(threshold=0.9)],
            backend=JevBackend(client=client, timeout=5, max_retries=0),
            calibration=config(),
        ).load_calibration(artifact)
        result = await restored.aevaluate_one(EvaluationSample(input="new", response="0.8"))
    assert result.metrics["answer_relevancy"].calibrated_probability == pytest.approx(0.6)
    assert result.passed is False


@pytest.mark.parametrize("version", [1, 2, 3])
async def test_jev_artifacts_without_historical_endpoint_require_refit(tmp_path, version):
    requests = []
    async with jev_client(requests=requests) as client:
        pipeline = await fitted_jev(client)
        bundle = pipeline.calibration_bundle
        data = (
            bundle.model_dump(mode="json")
            if version == 3
            else legacy_artifact_data(bundle, version=version)
        )
        if version >= 2:
            configuration = data["backend_provenance"]["configuration"]
            del configuration["endpoint"]
            configuration["compiler_version"] = "1"
            data["backend_provenance"]["fingerprint"] = digest(configuration)
        artifact = tmp_path / "old-jev.json"
        serialized = json.dumps(data)
        artifact.write_text(serialized, encoding="utf-8")
        calls = len(requests)
        with pytest.raises(CalibrationMismatchError, match="refit"):
            EvaluationPipeline(
                backend=JevBackend(client=client), calibration=config()
            ).load_calibration(artifact)
        assert len(requests) == calls
        restored = CalibrationBundle.load(artifact)
        assert restored.backend_provenance == data.get("backend_provenance")
        assert artifact.read_text(encoding="utf-8") == serialized
