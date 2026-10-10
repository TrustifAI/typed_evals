"""Calibration migration and optional backend identity remain provider independent."""

import json

import pytest
from conftest import FakeBackend, calibrated_rows

from typed_evals import (
    AnswerRelevancy,
    CalibrationBundle,
    CalibrationConfig,
    EvaluationPipeline,
    EvaluationSample,
    JevBackend,
    Metric,
)
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


def test_v1_custom_and_jev_workflows_load_without_fabricating_identity(scoring_backend, tmp_path):
    pipeline = fitted(scoring_backend)
    path = tmp_path / "legacy.json"
    legacy_artifact(pipeline.calibration_bundle, path)
    bundle = CalibrationBundle.load(path)
    assert bundle.schema_version == 1
    assert bundle.backend_provenance is None
    restored = EvaluationPipeline(backend=scoring_backend, calibration=config())
    restored.load_calibration(path)
    assert restored.evaluate_one(EvaluationSample(input="new", response="0.8")).passed
    jev = JevBackend(model=scoring_backend.model, client=object())
    restored_jev = EvaluationPipeline(backend=jev, calibration=config())
    restored_jev.load_calibration(path)
    assert restored_jev.calibration_bundle.backend_provenance is None


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


def test_jev_provenance_excludes_transport_credentials_and_thresholds():
    first = AnswerRelevancy(threshold=0.5)
    changed = AnswerRelevancy(threshold=0.9)
    original = backend_provenance(JevBackend(), {first.name: first.question()})
    other = backend_provenance(
        JevBackend(api_key="secret", timeout=5.0, max_retries=0, client=object()),
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
