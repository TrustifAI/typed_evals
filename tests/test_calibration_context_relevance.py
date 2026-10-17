"""Calibrate the retrieval fraction against the application's binary acceptance labels."""

import pytest
from conftest import FakeBackend

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    CalibrationExample,
    ContextRelevance,
    EvaluationPipeline,
    EvaluationSample,
)
from typed_evals.errors import CalibrationError, CalibrationMismatchError, DataLeakageError


def fraction_rows(prefix, *, empty_only=False):
    return [
        CalibrationExample(
            sample=EvaluationSample(
                input=f"{prefix}-{index}",
                response="answer",
                contexts=[] if empty_only or index % 2 == 0 else ["A useful passage."],
            ),
            labels={"context_relevance": index % 2},
        )
        for index in range(8)
    ]


def fraction_pipeline(backend, **metric_options):
    return EvaluationPipeline(
        [ContextRelevance(**metric_options)],
        backend=backend,
        calibration=CalibrationConfig(
            enabled=True,
            algorithm="isotonic",
            min_samples=4,
            min_validation_samples=2,
            min_class_samples=1,
        ),
    )


def omit_model_identity(backend):
    original_judge = backend.judge

    async def judge(state, questions):
        response = await original_judge(state, questions)
        # Deliberately bypass the provider response schema to exercise fit's
        # defensive check for a faulty custom backend.
        return response.model_copy(update={"model": None})

    backend.judge = judge


def test_fraction_calibration_includes_empty_retrieval_without_inventing_model_identity(backend):
    pipeline = fraction_pipeline(backend)
    report = pipeline.fit(fraction_rows("train"), validation_data=fraction_rows("holdout"))
    metric = report.metrics["context_relevance"]
    assert metric.raw_kind == "relevance_fraction"
    assert metric.training_samples == metric.validation_samples == 8
    assert any("numeric probability proxy" in note for note in metric.notes)
    assert any("metric-pass event" in note for note in metric.notes)
    assert pipeline.calibration_bundle.observed_model == backend.actual_model
    # Half the rows have known-empty retrieval, which does not require a judge.
    assert len(backend.calls) == 8
    before = len(backend.calls)
    results = pipeline.evaluate(
        [
            EvaluationSample(input="new empty", response="", contexts=[]),
            EvaluationSample(input="new useful", response="", contexts=["A useful passage."]),
        ]
    ).results
    assert len(backend.calls) == before + 1
    empty, useful = results
    assert empty.model is None
    assert empty.metrics["context_relevance"].raw_kind == "relevance_fraction"
    assert empty.metrics["context_relevance"].raw_score == 0
    assert empty.metrics["context_relevance"].calibrated_probability == 0
    assert useful.model == backend.actual_model
    assert useful.metrics["context_relevance"].raw_score == 1
    assert useful.metrics["context_relevance"].calibrated_probability == 1


def test_all_empty_fit_requires_at_least_one_observed_judge_model(backend):
    pipeline = fraction_pipeline(backend)
    with pytest.raises(CalibrationError, match="one observed model version"):
        pipeline.fit(
            fraction_rows("train", empty_only=True),
            validation_data=fraction_rows("holdout", empty_only=True),
        )
    assert not backend.calls
    assert pipeline.calibration_bundle is None


def test_nonempty_fraction_fit_rejects_missing_model_identity(backend):
    omit_model_identity(backend)
    pipeline = fraction_pipeline(backend)
    with pytest.raises(CalibrationError, match="one observed model version"):
        pipeline.fit(fraction_rows("train"), validation_data=fraction_rows("holdout"))
    assert pipeline.calibration_bundle is None


def test_empty_retrieval_does_not_hide_missing_model_for_other_panel_metrics(backend):
    omit_model_identity(backend)

    def rows(prefix):
        return [
            CalibrationExample(
                sample=row.sample,
                labels={**row.labels, "answer_relevancy": index % 2},
            )
            for index, row in enumerate(fraction_rows(prefix, empty_only=True))
        ]

    pipeline = EvaluationPipeline(
        [ContextRelevance(), AnswerRelevancy()],
        backend=backend,
        calibration=CalibrationConfig(
            enabled=True, min_samples=4, min_validation_samples=2, min_class_samples=1
        ),
    )
    with pytest.raises(CalibrationError, match="one observed model version"):
        pipeline.fit(rows("train"), validation_data=rows("holdout"))
    assert pipeline.calibration_bundle is None


def test_fraction_fit_rejects_different_observed_models_across_samples():
    backend = FakeBackend()

    def answer(state, questions):
        backend.actual_model = "judge-a" if state["input"].startswith("train") else "judge-b"
        return {name: {"type": "noul", "noul": 0.8} for name in questions}

    backend.answer = answer
    pipeline = fraction_pipeline(backend)
    with pytest.raises(CalibrationError, match="one observed model version"):
        pipeline.fit(fraction_rows("train"), validation_data=fraction_rows("holdout"))
    assert pipeline.calibration_bundle is None


@pytest.mark.parametrize(
    "options",
    [{"granularity": "passage"}, {"unit_threshold": 0.6}, {"questions_per_call": 16}],
)
def test_fraction_execution_configuration_changes_require_refitting(backend, options):
    pipeline = fraction_pipeline(backend)
    pipeline.fit(fraction_rows("train"), validation_data=fraction_rows("holdout"))
    with pytest.raises(CalibrationMismatchError, match="Metric wording"):
        pipeline.calibration_bundle.validate_for([ContextRelevance(**options)], backend.model)
    # The application decision threshold does not alter the signal or target.
    pipeline.calibration_bundle.validate_for([ContextRelevance(threshold=0.9)], backend.model)


def test_fraction_calibration_duplicate_guard_uses_original_selected_evidence(backend):
    training, holdout = fraction_rows("train"), fraction_rows("holdout")
    holdout[0] = CalibrationExample(
        sample=training[0].sample.model_copy(
            update={"response": "different unused answer", "id": "different-id"}
        ),
        labels={"context_relevance": 1},
    )
    pipeline = fraction_pipeline(backend)
    with pytest.raises(DataLeakageError, match="repeated sample content"):
        pipeline.fit(training, validation_data=holdout)
    assert not backend.calls
