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
    return CalibrationConfig(
        enabled=True, algorithm="isotonic", min_samples=20, min_validation_samples=10, **kwargs
    )


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
        pytest.param([10**400, 0.8], [0, 1], id="huge-positive-integer"),
        pytest.param([-(10**400), 0.8], [0, 1], id="huge-negative-integer"),
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


def test_probability_diagnostics_describe_errors_at_the_actual_threshold():
    stats = probability_diagnostics(
        [0.2, 0.5, 0.7, 0.7, 0.9, 0.95], [0, 1, 0, 1, 0, 1], threshold=0.7
    )
    operating = stats.at_threshold
    assert operating.threshold == 0.7
    # Equality passes, as it does in evaluation. The 0.5-positive is now rejected.
    assert operating.true_positives == operating.false_positives == 2
    assert operating.true_negatives == operating.false_negatives == 1
    assert operating.accuracy == pytest.approx(0.5)
    assert stats.accuracy_at_half == pytest.approx(4 / 6)
    assert operating.false_acceptance_rate == pytest.approx(2 / 3)
    assert operating.false_rejection_rate == pytest.approx(1 / 3)
    assert operating.precision == pytest.approx(0.5)
    assert operating.recall == pytest.approx(2 / 3)
    assert operating.balanced_accuracy == pytest.approx(0.5)


@pytest.mark.parametrize(
    "predictions,labels,expected",
    [
        (
            [0.8, 0.9],
            [0, 0],
            {
                "recall": None,
                "false_rejection_rate": None,
                "precision": 0,
                "false_acceptance_rate": 1,
                "balanced_accuracy": None,
            },
        ),
        (
            [0.1, 0.2],
            [0, 1],
            {
                "precision": None,
                "recall": 0,
                "false_acceptance_rate": 0,
                "false_rejection_rate": 1,
                "balanced_accuracy": 0.5,
            },
        ),
        (
            [0.8, 0.9],
            [1, 1],
            {
                "false_acceptance_rate": None,
                "recall": 1,
                "false_rejection_rate": 0,
                "balanced_accuracy": None,
            },
        ),
    ],
)
def test_threshold_rates_report_undefined_denominators_as_unavailable(
    predictions, labels, expected
):
    operating = probability_diagnostics(predictions, labels).at_threshold
    for name, value in expected.items():
        assert getattr(operating, name) == value


@pytest.mark.parametrize("threshold", [True, "0.5", None, -0.1, 1.1, math.nan, 10**400])
def test_probability_diagnostics_rejects_invalid_thresholds(threshold):
    with pytest.raises(CalibrationError, match="threshold"):
        probability_diagnostics([0.2, 0.8], [0, 1], threshold=threshold)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(10**400, id="huge-positive-integer"),
        pytest.param(-(10**400), id="huge-negative-integer"),
    ],
)
def test_probability_diagnostics_rejects_huge_scores_without_overflow(value):
    with pytest.raises(CalibrationError, match="raw scores must be finite values"):
        probability_diagnostics([value, 0.8], [0, 1])


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
    assert metrics.raw_kind == "event_probability"
    assert metrics.raw.brier == pytest.approx(0.28)
    assert metrics.calibrated.brier == pytest.approx(0.24)
    assert metrics.brier_improved
    assert metrics.constant_prior_brier == pytest.approx(0.25)
    assert metrics.raw.at_threshold.threshold == metrics.calibrated.at_threshold.threshold == 0.7
    assert metrics.raw.at_threshold.true_positives == 12
    assert metrics.raw.at_threshold.false_positives == 8
    assert metrics.raw.at_threshold.false_negatives == 8
    assert metrics.raw.at_threshold.true_negatives == 12
    assert metrics.calibrated.at_threshold.true_positives == 0
    assert metrics.calibrated.at_threshold.false_positives == 0
    assert metrics.calibrated.at_threshold.false_negatives == 20
    assert metrics.calibrated.at_threshold.true_negatives == 20
    assert metrics.calibrated.at_threshold.false_acceptance_rate == 0
    assert metrics.calibrated.at_threshold.false_rejection_rate == 1
    assert pipeline.calibration_bundle.curves["answer_relevancy"].n_samples == 40
    result = pipeline.evaluate_one(EvaluationSample(input="new production query", response="0.8"))
    metric = result.metrics["answer_relevancy"]
    assert metric.raw_score == 0.8
    assert metric.calibrated_probability == pytest.approx(0.6)
    assert metric.calibration_target == "metric_pass"
    assert metric.passed is False  # Threshold 0.7 uses calibrated 0.6, not raw 0.8.


def test_constant_prior_baseline_uses_training_prevalence_without_holdout_leakage(scoring_backend):
    validation = calibrated_rows("val")
    validation[:10] = [
        CalibrationExample(sample=row.sample, labels={"answer_relevancy": 1})
        for row in validation[:10]
    ]
    pipeline = EvaluationPipeline(backend=scoring_backend, calibration=config())
    report = pipeline.fit(calibrated_rows(), validation_data=validation)
    metric = report.metrics["answer_relevancy"]
    assert metric.training_positive_rate == 0.5
    assert metric.constant_prior_brier == pytest.approx(0.25)
    validation_prevalence = sum(row.labels["answer_relevancy"] for row in validation) / len(
        validation
    )
    assert validation_prevalence != metric.training_positive_rate
    assert metric.constant_prior_brier != pytest.approx(
        validation_prevalence * (1 - validation_prevalence)
    )


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
    report = pipeline.fit(data)
    assert report.split == "automatic_group_holdout"
    stats = report.metrics["answer_relevancy"]
    assert (stats.training_samples, stats.validation_samples) == (80, 20)
    # Fit submits the training rows followed by the independent holdout. Observe
    # the actual requests, so a correct private splitter used incorrectly by fit
    # cannot make this test pass.
    inputs = [state["input"] for state, _ in scoring_backend.calls]
    assert len(inputs) == len(set(inputs)) == len(data)
    assert set(inputs) == {row.sample.input for row in data}
    train_groups = {value.split("-")[0] for value in inputs[:80]}
    validation_groups = {value.split("-")[0] for value in inputs[80:]}
    assert len(train_groups) == 8
    assert len(validation_groups) == 2
    assert train_groups.isdisjoint(validation_groups)

    previous_curves = pipeline.calibration_bundle.curves
    scoring_backend.calls.clear()
    repeated_report = pipeline.fit(data)
    assert [state["input"] for state, _ in scoring_backend.calls] == inputs
    assert repeated_report == report
    assert pipeline.calibration_bundle.curves == previous_curves

    # Both halves are reserved: held-out rows must not later become test data.
    by_input = {row.sample.input: row.sample for row in data}
    for reserved_input in (inputs[0], inputs[80]):
        with pytest.raises(DataLeakageError, match="reuses calibration"):
            pipeline.evaluate_one(by_input[reserved_input])
    assert len(scoring_backend.calls) == len(data)


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


def test_failed_refit_closes_session_preserves_predictions_and_allows_retry(scoring_backend):
    pipeline, _ = fitted_pipeline(scoring_backend)
    previous = pipeline.calibration_bundle
    answer = scoring_backend.answer

    def fail_during_judging(state, questions):
        raise RuntimeError("provider unavailable during refit")

    scoring_backend.answer = fail_during_judging
    with pytest.raises(RuntimeError, match="provider unavailable during refit"):
        pipeline.fit(calibrated_rows("refit"), validation_data=calibrated_rows("refit-holdout"))
    assert pipeline.calibration_bundle is previous
    assert scoring_backend.sessions == scoring_backend.closed == 2
    assert scoring_backend.active == 0

    scoring_backend.answer = answer
    metric = pipeline.evaluate_one(EvaluationSample(input="fresh", response="0.8")).metrics[
        "answer_relevancy"
    ]
    assert metric.calibrated_probability == pytest.approx(0.6)
    assert metric.passed is False
    pipeline.fit(calibrated_rows("retry"), validation_data=calibrated_rows("retry-holdout"))
    assert pipeline.calibration_bundle is not previous
    assert scoring_backend.sessions == scoring_backend.closed == 4


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


@pytest.mark.parametrize("version", [1, 2, 3])
def test_old_artifacts_load_without_inventing_threshold_or_baseline_diagnostics(
    scoring_backend, tmp_path, version
):
    pipeline, _ = fitted_pipeline(scoring_backend)
    payload = pipeline.calibration_bundle.model_dump(mode="json")
    payload["schema_version"] = version
    metric = payload["report"]["metrics"]["answer_relevancy"]
    del metric["raw_kind"]
    del metric["constant_prior_brier"]
    del metric["raw"]["at_threshold"]
    del metric["calibrated"]["at_threshold"]
    if version < 3:
        del payload["curves"]["answer_relevancy"]["algorithm"]
        del metric["algorithm"]
        del metric["log_loss_improved"]
    if version == 1:
        del payload["backend_provenance"]
    path = tmp_path / f"legacy-v{version}.json"
    path.write_text(json.dumps(payload))
    loaded = CalibrationBundle.load(path)
    restored = loaded.report.metrics["answer_relevancy"]
    assert restored.raw_kind is None
    assert restored.constant_prior_brier is None
    assert restored.raw.at_threshold is restored.calibrated.at_threshold is None
    assert restored.raw.accuracy_at_half == metric["raw"]["accuracy_at_half"]
    if version < 3:
        saved = loaded.model_dump(mode="json")["report"]["metrics"]["answer_relevancy"]
        assert "constant_prior_brier" not in saved
        assert "raw_kind" not in saved
        assert "at_threshold" not in saved["raw"]
        assert "at_threshold" not in saved["calibrated"]


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
