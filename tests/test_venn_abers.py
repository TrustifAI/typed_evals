"""Exact IVAP reference checks and size-adaptive pipeline behavior."""

import builtins
import json
import math
import random

import numpy as np
import pytest
from conftest import calibrated_rows
from pydantic import ValidationError
from sklearn.isotonic import IsotonicRegression

from typed_evals import (
    AnswerRelevancy,
    CalibrationBundle,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    IsotonicCalibrator,
    Metric,
    VennAbersCalibrator,
)
from typed_evals.calibration import MetricCalibrationReport, probability_diagnostics
from typed_evals.errors import CalibrationError


@pytest.mark.parametrize("seed", range(20))
def test_venn_pair_matches_two_augmented_isotonic_fits_with_ties_and_boundaries(seed):
    rng = random.Random(seed)
    scores = [round(rng.uniform(0.1, 0.9), 1) for _ in range(80)]
    labels = [rng.randint(0, 1) for _ in scores]
    curve = VennAbersCalibrator.fit(scores, labels)
    grid = sorted({0.0, 1.0, *scores, *(i / 40 for i in range(41))})
    predictions = []
    for score in grid:
        expected = tuple(
            float(
                IsotonicRegression(out_of_bounds="clip")
                .fit([*scores, score], [*labels, label])
                .predict([score])[0]
            )
            for label in (0, 1)
        )
        p0, p1 = curve.predict_interval(score)
        assert (p0, p1) == pytest.approx(expected, abs=1e-12)
        probability = curve.predict(score)
        assert probability == pytest.approx(p1 / (1 - p0 + p1))
        assert 0 < probability < 1
        predictions.append(probability)
    assert all(a <= b for a, b in zip(predictions, predictions[1:], strict=False))


def test_venn_avoids_endpoint_certainty_on_separated_small_training_set():
    scores, labels = [0.2] * 40 + [0.8] * 40, [0] * 40 + [1] * 40
    isotonic = IsotonicCalibrator.fit(scores, labels)
    venn = VennAbersCalibrator.fit(scores, labels)
    assert isotonic.predict(0) == 0
    assert isotonic.predict(1) == 1
    assert venn.predict(0) == pytest.approx(1 / 42)
    assert venn.predict(1) == pytest.approx(41 / 42)
    assert math.isfinite(-math.log(venn.predict(0)))
    assert math.isfinite(-math.log1p(-venn.predict(1)))


def test_constant_signal_uses_smoothed_base_rate_at_observed_score():
    curve = VennAbersCalibrator.fit(np.array([0.4] * 4), np.array([0, 1, 1, 1]))
    assert curve.predict_interval(np.float64(0.4)) == pytest.approx((3 / 5, 4 / 5))
    assert curve.predict(0.4) == pytest.approx(4 / 6)
    assert curve.predict_interval(0) == pytest.approx((0, 4 / 5))
    assert curve.predict_interval(1) == pytest.approx((3 / 5, 1))


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([], []),
        ([0.1], [0, 1]),
        ([0.1, math.nan], [0, 1]),
        ([0.1, 0.9], [1, 1]),
        ([True, 0.9], [0, 1]),
        ([0.1, 0.9], [0, 0.5]),
    ],
)
def test_venn_invalid_training_values_are_rejected(scores, labels):
    with pytest.raises(CalibrationError):
        VennAbersCalibrator.fit(scores, labels)


@pytest.mark.parametrize("score", [True, None, "0.5", math.nan, math.inf, -0.1, 1.1])
def test_venn_invalid_prediction_values_are_rejected(score):
    curve = VennAbersCalibrator.fit([0.1, 0.9], [0, 1])
    with pytest.raises(CalibrationError):
        curve.predict(score)
    with pytest.raises(CalibrationError):
        curve.predict_interval(score)


@pytest.mark.parametrize(
    "changes",
    [
        {"x": []},
        {"x": [0.9, 0.1]},
        {"p0": [0.0]},
        {"p1": [1.0, 0.5]},
        {"p0": [0.8, 0.7]},
        {"p0": [0, 1]},
        {"p1": [0, 1]},
        {"n_samples": 1},
        {"x": [0.1, 0.5, 0.9], "p0": [0, 0.1, 0.2], "p1": [0.3, 0.4, 0.5]},
    ],
)
def test_corrupt_venn_tables_are_rejected(changes):
    data = {"x": [0.1, 0.9], "p0": [0, 0.5], "p1": [0.5, 1], "n_samples": 2}
    with pytest.raises(ValidationError):
        VennAbersCalibrator(**{**data, **changes})


@pytest.mark.parametrize("training_count,expected", [(1999, "venn_abers"), (2000, "isotonic")])
def test_auto_uses_labeled_training_count_at_2000_boundary(
    scoring_backend, training_count, expected
):
    pipeline = EvaluationPipeline(
        backend=scoring_backend, calibration=CalibrationConfig(enabled=True)
    )
    report = pipeline.fit(
        calibrated_rows(size=training_count), validation_data=calibrated_rows("validation")
    )
    assert report.metrics["answer_relevancy"].algorithm == expected
    assert pipeline.calibration_bundle.curves["answer_relevancy"].algorithm == expected
    assert pipeline.calibration_bundle.curves["answer_relevancy"].n_samples == training_count


def test_auto_selects_per_metric_after_sparse_labels_and_holdout(scoring_backend):
    second = Metric(
        name="second", instructions="Is the response acceptable?", pass_definition="Acceptable"
    )
    rows = []
    for row in calibrated_rows(size=200):
        index = int(row.sample.input.split("-")[-1])
        labels = {"answer_relevancy": row.labels["answer_relevancy"]}
        if index % 4 < 2:
            labels["second"] = row.labels["answer_relevancy"]
        rows.append(CalibrationExample(sample=row.sample, labels=labels))
    pipeline = EvaluationPipeline(
        [AnswerRelevancy(), second],
        backend=scoring_backend,
        calibration=CalibrationConfig(
            enabled=True, min_samples=20, min_validation_samples=10, isotonic_min_samples=120
        ),
    )
    report = pipeline.fit(rows)
    assert report.metrics["answer_relevancy"].training_samples == 160
    assert report.metrics["second"].training_samples < 120
    assert report.metrics["answer_relevancy"].algorithm == "isotonic"
    assert report.metrics["second"].algorithm == "venn_abers"
    assert len(scoring_backend.calls) == 200


@pytest.mark.parametrize("algorithm", ["venn_abers", "isotonic"])
def test_user_algorithm_override_and_roundtrip_preserve_predictions(
    scoring_backend, tmp_path, algorithm
):
    config = CalibrationConfig(
        enabled=True,
        algorithm=algorithm,
        min_samples=20,
        min_validation_samples=10,
        isotonic_min_samples=2,
    )
    pipeline = EvaluationPipeline(backend=scoring_backend, calibration=config)
    report = pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    stats = report.metrics["answer_relevancy"]
    assert stats.algorithm == algorithm
    assert stats.log_loss_improved == (stats.calibrated.log_loss < stats.raw.log_loss)
    path = tmp_path / "calibration.json"
    pipeline.save_calibration(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 3
    assert payload["curves"]["answer_relevancy"]["algorithm"] == algorithm
    restored = EvaluationPipeline(
        backend=scoring_backend, calibration=CalibrationConfig(enabled=True)
    )
    restored.load_calibration(path)
    curve = pipeline.calibration_bundle.curves["answer_relevancy"]
    sample = EvaluationSample(input="new-query", response="0.8")
    assert restored.evaluate_one(sample).metrics["answer_relevancy"].score == curve.predict(0.8)
    assert restored.calibration_bundle == CalibrationBundle.load(path)


def test_log_loss_regression_is_reported(scoring_backend):
    # Isotonic maps the perfectly separated training scores to hard 0/1, then
    # receives both classes at each score on an independent holdout.
    train = [
        CalibrationExample(
            sample=row.sample, labels={"answer_relevancy": int(row.sample.response == "0.8")}
        )
        for row in calibrated_rows()
    ]
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, algorithm="isotonic", min_samples=20),
    )
    stats = pipeline.fit(train, validation_data=calibrated_rows("validation")).metrics[
        "answer_relevancy"
    ]
    assert not stats.log_loss_improved
    assert any("log loss did not improve" in note for note in stats.notes)


def test_venn_fitting_and_inference_need_no_sklearn(scoring_backend, monkeypatch):
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("optional dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, min_samples=20),
    )
    pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    assert pipeline.evaluate_one(EvaluationSample(input="fresh", response="0.8")).passed
    calls_before = len(scoring_backend.calls)
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, algorithm="isotonic", min_samples=20),
    )
    with pytest.raises(CalibrationError, match="dependencies"):
        pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    assert len(scoring_backend.calls) == calls_before


@pytest.mark.parametrize(
    "changes",
    [
        {"algorithm": "platt"},
        {"isotonic_min_samples": True},
        {"isotonic_min_samples": 1},
        {"isotonic_min_samples": 2000.0},
    ],
)
def test_invalid_algorithm_configuration_rejected(changes):
    with pytest.raises(ValidationError):
        CalibrationConfig(**changes)


def test_legacy_report_infers_missing_log_loss_improvement_from_diagnostics():
    report = MetricCalibrationReport(
        training_samples=100,
        validation_samples=20,
        training_positive_rate=0.5,
        raw=probability_diagnostics([0.1, 0.9], [1, 0]),
        calibrated=probability_diagnostics([0.5, 0.5], [1, 0]),
        brier_improved=True,
    )
    assert report.log_loss_improved
    payload = report.model_dump(mode="json")
    del payload["log_loss_improved"]
    assert MetricCalibrationReport.model_validate(payload).log_loss_improved


@pytest.mark.parametrize("damage", ["missing_tag", "wrong_report_algorithm", "legacy_venn"])
def test_algorithm_metadata_corruption_rejected(scoring_backend, tmp_path, damage):
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, min_samples=20),
    )
    pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    data = pipeline.calibration_bundle.model_dump(mode="json")
    if damage == "missing_tag":
        del data["curves"]["answer_relevancy"]["algorithm"]
    elif damage == "wrong_report_algorithm":
        data["report"]["metrics"]["answer_relevancy"]["algorithm"] = "isotonic"
    else:
        data["schema_version"] = 2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CalibrationError, match="artifact"):
        CalibrationBundle.load(path)


@pytest.mark.parametrize("version", [1, 2])
def test_resaving_legacy_bundle_keeps_legacy_curve_and_report_schema(
    scoring_backend, tmp_path, version
):
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, algorithm="isotonic", min_samples=20),
    )
    pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    data = pipeline.calibration_bundle.model_dump(mode="json")
    data["schema_version"] = version
    if version == 1:
        del data["backend_provenance"]
    del data["curves"]["answer_relevancy"]["algorithm"]
    for name in ("algorithm", "log_loss_improved"):
        del data["report"]["metrics"]["answer_relevancy"][name]
    legacy = CalibrationBundle.model_validate(data)
    assert legacy.report.metrics["answer_relevancy"].log_loss_improved
    path = tmp_path / "legacy.json"
    legacy.save(path)
    exported = json.loads(path.read_text())
    assert set(exported["curves"]["answer_relevancy"]) == {"x", "y", "n_samples"}
    assert "algorithm" not in exported["report"]["metrics"]["answer_relevancy"]
    assert "log_loss_improved" not in exported["report"]["metrics"]["answer_relevancy"]
    restored = CalibrationBundle.load(path)
    assert restored.schema_version == version
    assert restored.curves["answer_relevancy"].predict(0.8) == pytest.approx(0.6)
