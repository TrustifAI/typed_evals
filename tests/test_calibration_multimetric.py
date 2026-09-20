"""Calibration target separation and sparse-label behavior across multiple metrics."""

import numpy as np
import pytest
from conftest import FakeBackend

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    IsotonicCalibrator,
    Metric,
)
from typed_evals.errors import CalibrationError, DataLeakageError


def test_each_metric_gets_its_own_curve_and_unlabeled_rows_are_excluded():
    second = Metric(
        name="second",
        instructions="Does `response` meet the second criterion?",
        pass_definition="Second criterion satisfied",
    )

    def rows(prefix):
        data = []
        for index in range(40):
            label = int(index % 2 == 1)
            labels = {"answer_relevancy": label}
            if index < 20:
                labels["second"] = 1 - label
            data.append(
                CalibrationExample(
                    sample=EvaluationSample(
                        input=f"{prefix}-{index}", response=str(0.8 if label else 0.2)
                    ),
                    labels=labels,
                )
            )
        return data

    backend = FakeBackend(
        lambda state, qs: {key: {"type": "noul", "noul": float(state["response"])} for key in qs}
    )
    pipeline = EvaluationPipeline(
        [AnswerRelevancy(), second],
        backend=backend,
        calibration=CalibrationConfig(enabled=True, min_samples=10, min_validation_samples=10),
    )
    report = pipeline.fit(rows("train"), validation_data=rows("val"))
    assert report.metrics["answer_relevancy"].training_samples == 40
    assert report.metrics["second"].training_samples == 20
    assert len(backend.calls) == 80  # Not 160 metric requests.
    first_curve = pipeline.calibration_bundle.curves["answer_relevancy"]
    second_curve = pipeline.calibration_bundle.curves["second"]
    assert first_curve.predict(0.8) == 1
    # A reversed signal cannot be repaired by an increasing isotonic map.
    assert second_curve.predict(0.8) == pytest.approx(0.5)
    assert all("labels" not in state for state, _ in backend.calls)


def test_isotonic_low_level_api_accepts_numpy_arrays():
    curve = IsotonicCalibrator.fit(np.array([0.1, 0.3, 0.7, 0.9]), np.array([0, 0, 1, 1]))
    assert curve.predict(np.float64(0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize("value", ["0.8", None, True])
def test_invalid_predict_values_are_explicit_calibration_errors(value):
    curve = IsotonicCalibrator(x=[0.1, 0.9], y=[0.2, 0.8], n_samples=20)
    with pytest.raises(CalibrationError):
        curve.predict(value)


def test_evaluation_group_overlap_requires_explicit_inspection_override(scoring_backend):
    from conftest import calibrated_rows

    train = calibrated_rows(group_id="train-group")
    val = calibrated_rows("val", group_id="val-group")
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, min_samples=20, min_validation_samples=10),
    )
    pipeline.fit(train, validation_data=val)
    sample = EvaluationSample(
        input="new content, related group", response="0.8", group_id="train-group"
    )
    with pytest.raises(DataLeakageError, match="group"):
        pipeline.evaluate([sample])
    assert pipeline.evaluate([sample], allow_calibration_overlap=True).calibrated


def test_changing_unused_fields_cannot_bypass_duplicate_guard(scoring_backend):
    from conftest import calibrated_rows

    train, validation = calibrated_rows(), calibrated_rows("val")
    pipeline = EvaluationPipeline(
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, min_samples=20, min_validation_samples=10),
    )
    pipeline.fit(train, validation_data=validation)
    changed = train[0].sample.model_copy(
        update={"id": "new-id", "reference": "irrelevant reference"}
    )
    with pytest.raises(DataLeakageError):
        pipeline.evaluate([changed])
    calls_before = len(scoring_backend.calls)
    validation[0] = CalibrationExample(sample=changed, labels={"answer_relevancy": 1})
    with pytest.raises(DataLeakageError):
        pipeline.fit(train, validation_data=validation)
    assert len(scoring_backend.calls) == calls_before
