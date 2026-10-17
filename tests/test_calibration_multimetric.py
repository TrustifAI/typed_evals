"""Calibration target separation and sparse-label behavior across multiple metrics."""

import numpy as np
import pytest
from conftest import FakeBackend

from typed_evals import (
    AnswerCorrectness,
    AnswerRelevancy,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    Faithfulness,
    IsotonicCalibrator,
    Metric,
)
from typed_evals.errors import CalibrationError, CalibrationMismatchError, DataLeakageError


def _complete_panel_rows(prefix):
    return [
        CalibrationExample(
            sample=EvaluationSample(
                input=f"{prefix}-{index}",
                response="Supported response",
                contexts=["Supporting passage"],
                reference="Expected answer",
            ),
            labels={"faithfulness": index % 2, "answer_correctness": index % 2},
        )
        for index in range(20)
    ]


@pytest.mark.parametrize("missing", ["raise", "skip"])
@pytest.mark.parametrize("absent", ["reference", "contexts"])
def test_calibrated_evaluation_cannot_shrink_the_actual_panel(backend, missing, absent):
    pipeline = EvaluationPipeline(
        [Faithfulness(), AnswerCorrectness()],
        backend=backend,
        missing=missing,
        errors="record",
        calibration=CalibrationConfig(enabled=True, min_samples=10, min_validation_samples=10),
    )
    pipeline.fit(_complete_panel_rows("train"), validation_data=_complete_panel_rows("validation"))
    complete = _complete_panel_rows("production")[0].sample
    incomplete = complete.model_copy(update={absent: None})
    calls_before, sessions_before = len(backend.calls), backend.sessions
    # A bad row later in the batch must prevent requests for the earlier good row too.
    with pytest.raises(CalibrationMismatchError, match="complete metric panel"):
        pipeline.evaluate([complete, incomplete])
    assert len(backend.calls) == calls_before
    assert backend.sessions == sessions_before


@pytest.mark.parametrize("split", ["training", "validation"])
@pytest.mark.parametrize("sparse_labels", [False, True])
def test_fitting_requires_a_complete_panel_even_on_unlabeled_metrics(backend, split, sparse_labels):
    training = _complete_panel_rows("train")
    validation = _complete_panel_rows("validation")
    rows = training if split == "training" else validation
    labels = {"faithfulness": 0} if sparse_labels else rows[0].labels
    rows[0] = CalibrationExample(
        sample=rows[0].sample.model_copy(update={"reference": None}), labels=labels
    )
    pipeline = EvaluationPipeline(
        [Faithfulness(), AnswerCorrectness()],
        backend=backend,
        missing="skip",
        calibration=CalibrationConfig(enabled=True, min_samples=10, min_validation_samples=10),
    )
    with pytest.raises(CalibrationError, match="answer_correctness lacks required sample evidence"):
        pipeline.fit(training, validation_data=validation)
    assert not backend.calls
    assert backend.sessions == 0


def test_uncalibrated_evaluation_still_allows_skipping_a_metric(backend):
    pipeline = EvaluationPipeline(
        [Faithfulness(), AnswerCorrectness()], backend=backend, missing="skip"
    )
    sample = _complete_panel_rows("production")[0].sample.model_copy(update={"reference": None})
    result = pipeline.evaluate_one(sample)
    assert result.metrics["faithfulness"].status == "ok"
    assert result.metrics["answer_correctness"].status == "skipped"
    assert set(backend.calls[0][1]) == {"faithfulness"}


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
        calibration=CalibrationConfig(
            enabled=True, algorithm="isotonic", min_samples=10, min_validation_samples=10
        ),
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
    assert report.metrics["second"].calibrated.brier == pytest.approx(
        report.metrics["second"].constant_prior_brier
    )
    assert any(
        "constant training-prior baseline" in note for note in report.metrics["second"].notes
    )
    assert all("labels" not in state for state, _ in backend.calls)


def test_isotonic_low_level_api_accepts_numpy_arrays():
    curve = IsotonicCalibrator.fit(np.array([0.1, 0.3, 0.7, 0.9]), np.array([0, 0, 1, 1]))
    assert curve.predict(np.float64(0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "value",
    [
        "0.8",
        None,
        True,
        pytest.param(10**400, id="huge-positive-integer"),
        pytest.param(-(10**400), id="huge-negative-integer"),
    ],
)
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
