import json
import math
import random

import numpy as np
import pytest
from conftest import calibrated_rows
from pydantic import ValidationError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from typed_evals import (
    AnswerRelevancy,
    CalibrationBundle,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    IsotonicCalibrator,
)
from typed_evals.calibration import probability_diagnostics
from typed_evals.errors import CalibrationError, CalibrationMismatchError, DataLeakageError


def config(**kwargs):
    return CalibrationConfig(enabled=True, min_samples=20, min_validation_samples=10, **kwargs)


def fitted_pipeline(backend):
    pipeline = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.7)], backend=backend, calibration=config()
    )
    report = pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    return pipeline, report


@pytest.mark.parametrize("seed", range(5))
def test_exported_knots_match_sklearn_including_ties_and_clipping(seed):
    rng = random.Random(seed)
    x = [round(rng.uniform(0.1, 0.9), 1) for _ in range(100)]
    y = [rng.randint(0, 1) for _ in range(100)]
    curve = IsotonicCalibrator.fit(x, y)
    sklearn_model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
        x, y
    )
    grid = np.linspace(0, 1, 201)
    ours = [curve.predict(float(value)) for value in grid]
    assert np.allclose(ours, sklearn_model.predict(grid), atol=1e-12)
    assert all(a <= b for a, b in zip(ours, ours[1:], strict=False))


def test_constant_signal_learns_only_base_rate():
    curve = IsotonicCalibrator.fit([0.4] * 4, [0, 1, 1, 1])
    assert curve.predict(0) == curve.predict(1) == 0.75


@pytest.mark.parametrize(
    "x,y",
    [
        ([], []),
        ([0.1], [1, 0]),
        ([math.nan, 0.8], [0, 1]),
        ([0.2, 0.8], [1, 1]),
        ([0.2, 0.8], [0.2, 0.8]),
    ],
)
def test_invalid_training_values_rejected(x, y):
    with pytest.raises(CalibrationError):
        IsotonicCalibrator.fit(x, y)


@pytest.mark.parametrize(
    "x,y", [([0.8, 0.2], [0.2, 0.8]), ([0.2, 0.8], [0.8, 0.2]), ([], []), ([0.2], [0.2, 0.8])]
)
def test_invalid_serialized_curves_are_rejected(x, y):
    with pytest.raises(ValidationError):
        IsotonicCalibrator(x=x, y=y, n_samples=10)


def test_probability_diagnostics_match_reference_and_include_probability_one():
    p, y = [0, 0.1, 0.4, 0.8, 1], [0, 1, 0, 1, 1]
    stats = probability_diagnostics(p, y, n_bins=5)
    assert stats.brier == pytest.approx(brier_score_loss(y, p))
    assert stats.log_loss == pytest.approx(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)))
    assert sum(bucket.count for bucket in stats.bins) == len(y)
    assert stats.bins[-1].count == 2
    assert stats.ece == pytest.approx(0.3)


def test_opt_in_and_unfitted_modes_fail_before_requests(scoring_backend):
    off = EvaluationPipeline(backend=scoring_backend)
    with pytest.raises(CalibrationError, match="opt-in"):
        off.fit(calibrated_rows())
    on = EvaluationPipeline(backend=scoring_backend, calibration=config())
    with pytest.raises(CalibrationError, match="unfitted"):
        on.evaluate([EvaluationSample(input="new", response="0.8")])
    assert not scoring_backend.calls


def test_explicit_holdout_improvement_and_calibrated_threshold(scoring_backend):
    pipeline, report = fitted_pipeline(scoring_backend)
    metrics = report.metrics["answer_relevancy"]
    assert metrics.raw.brier == pytest.approx(0.28)
    assert metrics.calibrated.brier == pytest.approx(0.24)
    assert metrics.brier_improved
    assert pipeline.calibration_bundle.curves["answer_relevancy"].n_samples == 40
    result = pipeline.evaluate_one(EvaluationSample(input="new production query", response="0.8"))
    metric = result.metrics["answer_relevancy"]
    assert metric.raw_score == 0.8
    assert metric.calibrated_probability == pytest.approx(0.6)
    assert metric.calibration_target == "metric_pass"
    assert metric.passed is False  # Threshold 0.7 uses calibrated 0.6, not raw 0.8.


def test_validation_labels_do_not_affect_fitted_curve(scoring_backend):
    train = calibrated_rows()
    validation = calibrated_rows("val")
    pipeline = EvaluationPipeline(backend=scoring_backend, calibration=config())
    pipeline.fit(train, validation_data=validation)
    original = pipeline.calibration_bundle.curves
    flipped = [
        CalibrationExample(
            sample=row.sample, labels={"answer_relevancy": 1 - row.labels["answer_relevancy"]}
        )
        for row in validation
    ]
    pipeline.fit(train, validation_data=flipped)
    assert pipeline.calibration_bundle.curves == original


def test_automatic_split_is_reproducible_and_keeps_groups_together(scoring_backend):
    data = []
    for group in range(10):
        data.extend(calibrated_rows(f"g{group}", size=10, group_id=f"group-{group}"))
    pipeline = EvaluationPipeline(backend=scoring_backend, calibration=config())
    train, validation = pipeline._split(tuple(data))
    assert pipeline._split(tuple(data)) == (train, validation)
    assert {row.sample.group_id for row in train}.isdisjoint(
        row.sample.group_id for row in validation
    )
    report = pipeline.fit(data)
    assert report.split == "automatic_group_holdout"
    assert report.metrics["answer_relevancy"].training_samples == 80


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "id_changed_duplicate",
        "overlapping_group",
        "one_class",
        "too_small",
        "unknown_label",
    ],
)
def test_bad_calibration_data_fails_before_network(scoring_backend, kind):
    train, validation = calibrated_rows(), calibrated_rows("validation")
    if kind == "duplicate":
        validation[0] = train[0]
    elif kind == "id_changed_duplicate":
        validation[0] = train[0].model_copy(
            update={"sample": train[0].sample.model_copy(update={"id": "new"})}
        )
    elif kind == "overlapping_group":
        train = calibrated_rows(group_id="g")
        validation = calibrated_rows("v", group_id="g")
    elif kind == "one_class":
        train = [
            CalibrationExample(sample=row.sample, labels={"answer_relevancy": 1}) for row in train
        ]
    elif kind == "too_small":
        train = train[:4]
    else:
        train[0] = CalibrationExample(sample=train[0].sample, labels={"typo": 1})
    with pytest.raises(CalibrationError):
        EvaluationPipeline(backend=scoring_backend, calibration=config()).fit(
            train, validation_data=validation
        )
    assert not scoring_backend.calls


def test_failed_refit_preserves_prior_model(scoring_backend):
    pipeline, _ = fitted_pipeline(scoring_backend)
    previous = pipeline.calibration_bundle
    with pytest.raises(CalibrationError):
        pipeline.fit(calibrated_rows(size=2))
    assert pipeline.calibration_bundle is previous
    assert not pipeline._fitting


def test_save_load_is_json_and_preserves_predictions_and_guardrails(scoring_backend, tmp_path):
    pipeline, _ = fitted_pipeline(scoring_backend)
    path = tmp_path / "calibration.json"
    pipeline.save_calibration(path)
    assert json.loads(path.read_text())["target"] == "metric_pass"
    restored = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.9)], backend=scoring_backend, calibration=config()
    )
    restored.load_calibration(path)
    sample = EvaluationSample(input="brand new", response="0.8")
    assert restored.evaluate_one(sample).metrics["answer_relevancy"].score == pytest.approx(0.6)
    with pytest.raises(DataLeakageError):
        restored.evaluate([calibrated_rows()[0].sample])


def test_changed_metric_and_changed_observed_model_reject_calibration(scoring_backend):
    pipeline, _ = fitted_pipeline(scoring_backend)
    changed = AnswerRelevancy().model_copy(update={"instructions": "a new question"})
    with pytest.raises(CalibrationMismatchError):
        pipeline.calibration_bundle.validate_for([changed], scoring_backend.model)
    with pytest.raises(CalibrationMismatchError):
        pipeline.calibration_bundle.validate_for(pipeline.evaluator.metrics, "new-model")
    scoring_backend.actual_model = "new-model"
    with pytest.raises(CalibrationMismatchError):
        pipeline.evaluate([EvaluationSample(input="new", response="0.8")])


def test_corrupt_artifact_fails_validation(scoring_backend, tmp_path):
    pipeline, _ = fitted_pipeline(scoring_backend)
    data = pipeline.calibration_bundle.model_dump(mode="json")
    data["curves"]["answer_relevancy"]["x"] = [0.8, 0.2]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CalibrationError, match="artifact"):
        CalibrationBundle.load(path)


@pytest.mark.parametrize("sync", [False, True])
async def test_automated_run_in_running_loop(scoring_backend, sync):
    pipeline = EvaluationPipeline(backend=scoring_backend, calibration=config())
    samples = [EvaluationSample(input="fresh", response="0.2")]
    options = {
        "calibration_data": calibrated_rows(),
        "validation_data": calibrated_rows("holdout"),
    }
    if sync:
        report = pipeline.run(samples, **options)
    else:
        report = await pipeline.arun(samples, **options)
    assert report.calibrated
    assert report.results[0].metrics["answer_relevancy"].score == pytest.approx(0.4)
