import json

import pytest
from conftest import FakeBackend, binary_answer, calibrated_rows

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    CalibrationExample,
    ContextRelevance,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    Faithfulness,
    Metric,
)
from typed_evals.errors import DataLeakageError, MissingInputError
from typed_evals.evaluation.validation import avalidate_metrics, validate_metrics


def rows(scores, labels):
    return [
        CalibrationExample(
            sample=EvaluationSample(input=f"test-{index}", response=str(score)),
            labels={"answer_relevancy": label},
        )
        for index, (score, label) in enumerate(zip(scores, labels, strict=True))
    ]


def test_validation_uses_configured_threshold_and_known_confusion(scoring_backend):
    evaluator = Evaluator([AnswerRelevancy(threshold=0.7)], backend=scoring_backend)
    report = validate_metrics(rows([0.9, 0.7, 0.6, 0.1], [1, 0, 1, 0]), evaluator=evaluator)
    metric = report.metrics["answer_relevancy"]
    assert metric.labeled == metric.evaluated == 4
    assert metric.unavailable == metric.skipped == metric.errors == 0
    assert metric.coverage == 1
    assert metric.raw == metric.effective
    assert metric.raw_kind == metric.effective_kind == "event_probability"
    decisions = metric.effective.at_threshold
    assert decisions.threshold == metric.threshold == 0.7
    assert decisions.true_positives == decisions.true_negatives == 1
    assert decisions.false_positives == decisions.false_negatives == 1
    assert decisions.accuracy == decisions.false_acceptance_rate == 0.5
    assert decisions.false_rejection_rate == decisions.balanced_accuracy == 0.5
    assert metric.raw_roc_auc == metric.effective_roc_auc == 0.75
    assert metric.majority_baseline_accuracy == 0.5
    assert report.examples == 4
    assert not report.calibrated
    assert report.label_source == "caller_supplied"
    assert all("labels" not in state for state, _ in scoring_backend.calls)


@pytest.mark.parametrize(
    "scores,labels,auc",
    [
        ([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], 0.5),
        ([0.1, 0.1, 0.9, 0.9], [0, 1, 0, 1], 0.5),
        ([0.1, 0.4, 0.9, 0.7], [1, 1, 0, 0], 0.0),
        ([0.1, 0.4, 0.9, 0.7], [0, 0, 1, 1], 1.0),
        ([0.1, 0.4, 0.9, 0.7], [1, 1, 1, 1], None),
        ([0.1, 0.4, 0.9, 0.7], [0, 0, 0, 0], None),
    ],
)
def test_auc_credits_ties_and_is_unavailable_for_single_class(scores, labels, auc, scoring_backend):
    metric = validate_metrics(
        rows(scores, labels), evaluator=Evaluator(backend=scoring_backend)
    ).metrics["answer_relevancy"]
    assert metric.raw_roc_auc == metric.effective_roc_auc == auc


def test_sparse_labels_preserve_panel_and_exclude_unlabeled_judgments(scoring_backend):
    second = Metric(name="second", instructions="Second?", pass_definition="Second criterion met")
    unlabeled = Metric(
        name="unlabeled", instructions="Third?", pass_definition="Third criterion met"
    )
    examples = rows([0.9, 0.2, 0.4], [1, 0, 1])
    examples[1] = examples[1].model_copy(update={"labels": {"second": 1}})
    report = validate_metrics(
        examples,
        evaluator=Evaluator([AnswerRelevancy(), second, unlabeled], backend=scoring_backend),
    )
    first = report.metrics["answer_relevancy"]
    assert first.labeled == first.evaluated == first.raw.count == 2
    assert first.raw.brier == pytest.approx((0.1**2 + 0.6**2) / 2)
    assert first.majority_baseline_accuracy == 1
    assert report.metrics["second"].labeled == report.metrics["second"].evaluated == 1
    assert report.metrics["second"].raw_roc_auc is None
    third = report.metrics["unlabeled"]
    assert third.labeled == third.evaluated == third.unavailable == 0
    assert (
        third.coverage is third.raw is third.effective is third.majority_baseline_accuracy is None
    )
    assert third.raw_roc_auc is third.effective_roc_auc is None
    assert len(scoring_backend.calls) == len(examples)
    assert all(set(questions) == set(report.metrics) for _, questions in scoring_backend.calls)


def test_unavailable_labeled_judgments_reduce_coverage_without_becoming_zero():
    def answer(state, questions):
        if state["response"] == "error":
            return {}
        return {name: binary_answer(question, 0.8) for name, question in questions.items()}

    backend = FakeBackend(answer)
    examples = [
        CalibrationExample(
            sample=EvaluationSample(
                input=kind, response=kind, contexts=None if kind == "skip" else ("evidence",)
            ),
            labels={"faithfulness": int(kind == "ok")},
        )
        for kind in ("skip", "error", "ok")
    ]
    metric = validate_metrics(
        examples,
        evaluator=Evaluator([Faithfulness()], backend=backend, missing="skip", errors="record"),
    ).metrics["faithfulness"]
    assert metric.labeled == 3
    assert metric.evaluated == metric.skipped == metric.errors == 1
    assert metric.unavailable == 2
    assert metric.coverage == pytest.approx(1 / 3)
    assert metric.raw.count == metric.effective.count == 1
    assert metric.effective.brier == pytest.approx(0.04)
    assert metric.effective.at_threshold.accuracy == metric.majority_baseline_accuracy == 1


def test_all_unavailable_has_no_probability_or_baseline_diagnostics(backend):
    examples = [
        CalibrationExample(
            sample=EvaluationSample(input="question", response="answer"), labels={"faithfulness": 0}
        )
    ]
    metric = validate_metrics(
        examples, evaluator=Evaluator([Faithfulness()], backend=backend, missing="skip")
    ).metrics["faithfulness"]
    assert metric.labeled == metric.unavailable == metric.skipped == 1
    assert metric.evaluated == metric.errors == 0
    assert metric.coverage == 0
    assert metric.raw is metric.effective is metric.majority_baseline_accuracy is None
    assert metric.raw_roc_auc is metric.effective_roc_auc is None
    assert backend.sessions == 0


def test_unknown_labels_preflight_entire_batch_before_judging(backend):
    examples = rows([0.1, 0.9], [0, 1])
    examples[1] = examples[1].model_copy(update={"labels": {"unknown": 1}})
    with pytest.raises(ValueError, match="Unknown metric labels"):
        validate_metrics(examples, evaluator=Evaluator(backend=backend))
    assert backend.calls == []
    assert backend.sessions == 0


def test_missing_labeled_evidence_preflights_before_judging(backend):
    examples = [
        CalibrationExample(
            sample=EvaluationSample(input=str(index), response="answer", contexts=contexts),
            labels={"faithfulness": 1},
        )
        for index, contexts in enumerate((("evidence",), None))
    ]
    with pytest.raises(MissingInputError, match="Validation row 1: faithfulness"):
        validate_metrics(examples, evaluator=Evaluator([Faithfulness()], backend=backend))
    assert backend.calls == []
    assert backend.sessions == 0


@pytest.mark.parametrize(
    "duplicate_labels", [{"answer_relevancy": 0}, {"answer_relevancy": 1}, {"second": 1}]
)
def test_repeated_selected_evidence_rejects_duplicate_conflicting_and_sparse_labels(
    duplicate_labels, backend
):
    first = rows([0.2], [0])[0]
    duplicate = CalibrationExample(
        sample=first.sample.model_copy(
            update={
                "id": "different-id",
                "group_id": "different-group",
                "metadata": {"source": "different-source"},
                "reference": "unused reference does not change selected evidence",
            }
        ),
        labels=duplicate_labels,
    )
    second = Metric(name="second", instructions="Second?", pass_definition="Second criterion met")
    with pytest.raises(DataLeakageError, match="repeated selected-panel evidence"):
        validate_metrics(
            [first, duplicate], evaluator=Evaluator([AnswerRelevancy(), second], backend=backend)
        )
    assert backend.calls == []
    assert backend.sessions == 0


def test_known_empty_retrieval_is_evaluated_in_validation():
    def answer(state, questions):
        assert state["contexts"] == []
        return {name: binary_answer(question, 0.0) for name, question in questions.items()}

    backend = FakeBackend(answer)
    example = CalibrationExample(
        sample=EvaluationSample(input="question", response="unsupported fact", contexts=[]),
        labels={"faithfulness": 0},
    )
    metric = validate_metrics(
        [example], evaluator=Evaluator([Faithfulness()], backend=backend)
    ).metrics["faithfulness"]
    assert metric.evaluated == metric.labeled == 1
    assert metric.unavailable == 0
    assert metric.coverage == metric.effective.at_threshold.accuracy == 1


def test_ordinal_validation_reports_proxy_semantics_and_calibrated_probability():
    metric = Metric(
        name="quality",
        kind="score",
        instructions="How good is the answer?",
        criteria=("poor", "adequate", "excellent"),
        pass_definition="The answer meets the application's binary quality standard.",
    )

    def answer(state, questions):
        score = float(state["response"])
        return {
            name: {
                "type": "score",
                "score": score * 2,
                "confidence": 1,
                "probabilities": {"0": 1 - score, "1": 0, "2": score},
            }
            for name in questions
        }

    def examples(prefix):
        return [
            CalibrationExample(
                sample=EvaluationSample(input=f"{prefix}-{index}", response=str(score)),
                labels={"quality": label},
            )
            for index, (score, label) in enumerate([(0.2, 0), (0.4, 0), (0.6, 1), (0.8, 1)])
        ]

    pipeline = EvaluationPipeline(
        [metric],
        backend=FakeBackend(answer),
        calibration=CalibrationConfig(
            enabled=True, min_samples=4, min_validation_samples=2, min_class_samples=1
        ),
    )
    test = examples("test")
    raw_report = validate_metrics(test, evaluator=pipeline.evaluator)
    raw = raw_report.metrics["quality"]
    assert raw.raw_kind == raw.effective_kind == "normalized_ordinal_score"
    assert "numeric probability proxy" in raw_report.note
    assert raw.raw.brier == pytest.approx(0.1)
    pipeline.fit(examples("train"), validation_data=examples("holdout"))
    calibrated = validate_metrics(
        test, evaluator=pipeline.evaluator, calibration=pipeline.calibration_bundle
    ).metrics["quality"]
    assert calibrated.raw_kind == "normalized_ordinal_score"
    assert calibrated.effective_kind == "event_probability"


def test_fraction_validation_distinguishes_unit_fraction_from_event_probability():
    def answer(state, questions):
        return {
            name: {
                "type": "noul",
                "noul": 0.6
                if state["contexts"][question.instructions["target_context_index"]].startswith(
                    "Useful"
                )
                else 0.4,
            }
            for name, question in questions.items()
        }

    def examples(prefix):
        return [
            CalibrationExample(
                sample=EvaluationSample(
                    input=f"{prefix}-{index}",
                    response="unused",
                    contexts=[f"Useful fact {unit}." for unit in range(relevant)]
                    + [f"Noise {unit}." for unit in range(4 - relevant)]
                    if relevant
                    else [],
                ),
                labels={"context_relevance": int(relevant >= 2)},
            )
            for index, relevant in enumerate((0, 1, 3, 4))
        ]

    pipeline = EvaluationPipeline(
        [ContextRelevance()],
        backend=FakeBackend(answer),
        calibration=CalibrationConfig(
            enabled=True,
            algorithm="isotonic",
            min_samples=4,
            min_validation_samples=2,
            min_class_samples=1,
        ),
    )
    raw_report = validate_metrics(examples("test"), evaluator=pipeline.evaluator)
    raw = raw_report.metrics["context_relevance"]
    assert raw.raw_kind == raw.effective_kind == "relevance_fraction"
    assert "relevance_fraction" in raw_report.note
    assert "numeric probability proxy" in raw_report.note
    assert raw.evaluated == raw.labeled == 4
    assert raw.unavailable == 0
    assert raw.raw.brier == pytest.approx(0.03125)
    assert raw.raw_roc_auc == 1
    pipeline.fit(examples("train"), validation_data=examples("holdout"))
    calibrated = validate_metrics(
        examples("test"), evaluator=pipeline.evaluator, calibration=pipeline.calibration_bundle
    ).metrics["context_relevance"]
    assert calibrated.raw_kind == "relevance_fraction"
    assert calibrated.effective_kind == "event_probability"
    assert calibrated.raw.brier == pytest.approx(0.03125)
    assert calibrated.effective.brier == 0


def test_fraction_validation_rejects_repeated_selected_evidence(backend):
    first = CalibrationExample(
        sample=EvaluationSample(input="question", response="first", contexts=["A passage."]),
        labels={"context_relevance": 1},
    )
    duplicate = CalibrationExample(
        sample=first.sample.model_copy(update={"response": "different", "id": "different"}),
        labels={"context_relevance": 0},
    )
    with pytest.raises(DataLeakageError, match="repeated selected-panel evidence"):
        validate_metrics(
            [first, duplicate], evaluator=Evaluator([ContextRelevance()], backend=backend)
        )
    assert not backend.calls


@pytest.mark.parametrize("examples,error", [([], ValueError), ([{}], TypeError)])
def test_invalid_examples_preflight_without_judging(examples, error, backend):
    with pytest.raises(error, match="CalibrationExample"):
        validate_metrics(examples, evaluator=Evaluator(backend=backend))
    assert backend.sessions == 0


def test_calibration_is_applied_and_fit_holdout_overlap_remains_rejected(scoring_backend):
    pipeline = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.7)],
        backend=scoring_backend,
        calibration=CalibrationConfig(
            enabled=True, min_samples=4, min_validation_samples=2, min_class_samples=1
        ),
    )
    train, holdout = calibrated_rows(size=12), calibrated_rows("holdout", size=12)
    pipeline.fit(train, validation_data=holdout)
    bundle = pipeline.calibration_bundle
    examples = rows([0.2, 0.8], [0, 1])
    report = validate_metrics(examples, evaluator=pipeline.evaluator, calibration=bundle)
    metric = report.metrics["answer_relevancy"]
    assert report.calibrated
    assert metric.raw.brier != metric.effective.brier
    assert metric.raw.at_threshold.threshold == metric.effective.at_threshold.threshold == 0.7
    calls_before = len(scoring_backend.calls)
    for reserved in (train, holdout):
        with pytest.raises(DataLeakageError, match="reuses calibration/validation content"):
            validate_metrics(reserved, evaluator=pipeline.evaluator, calibration=bundle)
    assert len(scoring_backend.calls) == calls_before


async def test_async_and_sync_entrypoints_agree_inside_running_loop(scoring_backend):
    evaluator = Evaluator(backend=scoring_backend)
    examples = rows([0.1, 0.8], [0, 1])
    asynchronous = await avalidate_metrics(examples, evaluator=evaluator)
    synchronous = validate_metrics(examples, evaluator=evaluator)
    assert asynchronous == synchronous


def test_validation_report_serializes_diagnostics_and_label_provenance(scoring_backend, tmp_path):
    report = validate_metrics(
        rows([0.1, 0.8], [0, 1]), evaluator=Evaluator(backend=scoring_backend)
    )
    path = tmp_path / "validation.json"
    report.save(path)
    saved = json.loads(path.read_text())
    assert saved == report.to_dict()
    assert saved["label_source"] == "caller_supplied"
    assert saved["metrics"]["answer_relevancy"]["effective"]["at_threshold"]["accuracy"] == 1
