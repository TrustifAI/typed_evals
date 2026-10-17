"""Retrieval noise is scored over every source unit, with attributable failures."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import FakeBackend

from typed_evals import (
    AnswerRelevancy,
    ContextRelevance,
    EvaluationSample,
    Evaluator,
    Faithfulness,
    Metric,
)
from typed_evals.errors import DecisionRefusalError, InvalidAnswerError, MissingInputError


def unit_answers(state, questions):
    answers = {}
    for name, question in questions.items():
        target = question.instructions.get("target_context_index")
        probability = (
            (0.9 if state["contexts"][target].strip().startswith("Useful") else 0.1)
            if target is not None
            else 0.8
        )
        answers[name] = {"type": "noul", "noul": probability}
    return answers


def context_sample(contexts):
    return EvaluationSample(
        input="What information is useful?", response="Answer", contexts=contexts
    )


def test_one_relevant_passage_among_nine_noise_passages_scores_one_tenth():
    backend = FakeBackend(unit_answers)
    sample = context_sample(("Useful evidence.",) + tuple(f"Noise {i}." for i in range(9)))
    result = Evaluator([ContextRelevance()], backend=backend).evaluate_one(sample)

    metric = result.metrics["context_relevance"]
    assert metric.raw_kind == "relevance_fraction"
    assert metric.raw_score == metric.score == 0.1
    assert metric.passed is result.passed is False
    assert metric.details.total_units == metric.details.evaluated_units == 10
    assert metric.details.relevant_units == 1
    assert metric.details.unavailable_units == 0
    assert metric.details.coverage == 1
    assert len(backend.calls) == backend.sessions == backend.closed == 1


def test_noise_inside_a_relevant_passage_is_included_with_exact_source_spans():
    passage = "Useful answer. Noise follows!\nMore noise?"
    backend = FakeBackend(unit_answers)
    report = Evaluator([ContextRelevance()], backend=backend).evaluate([context_sample((passage,))])
    metric = report.results[0].metrics["context_relevance"]

    assert metric.score == pytest.approx(1 / 3)
    assert [passage[unit.start : unit.end] for unit in metric.details.units] == [
        "Useful answer. ",
        "Noise follows!",
        "\nMore noise?",
    ]
    assert [(unit.passage_index, unit.unit_index) for unit in metric.details.units] == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]
    assert metric.details.segmentation_version == "context-units-v1"
    assert passage not in json.dumps(report.to_dict())
    assert "Useful answer" not in json.dumps(report.to_dict())


def test_passage_granularity_is_an_explicit_coarser_denominator():
    backend = FakeBackend(unit_answers)
    metric = ContextRelevance(granularity="passage")
    result = Evaluator([metric], backend=backend).evaluate_one(
        context_sample(("Useful answer. Noise follows!", "Noise passage."))
    )
    details = result.metrics[metric.name].details
    assert result.metrics[metric.name].score == 0.5
    assert details.granularity == "passage"
    assert details.total_units == 2
    assert [(unit.unit_index, unit.start, unit.end) for unit in details.units] == [
        (0, 0, len("Useful answer. Noise follows!")),
        (0, 0, len("Noise passage.")),
    ]


@pytest.mark.parametrize("unit_threshold,expected", [(0.5, 2 / 3), (0.8, 1 / 3)])
def test_probability_cutoff_and_aggregate_acceptance_are_separate(unit_threshold, expected):
    probabilities = {"Low.": 0.49, "Borderline.": 0.5, "High.": 0.9}

    def answer(state, questions):
        return {
            name: {
                "type": "noul",
                "noul": probabilities[
                    state["contexts"][question.instructions["target_context_index"]]
                ],
            }
            for name, question in questions.items()
        }

    result = Evaluator(
        [ContextRelevance(threshold=0.3, unit_threshold=unit_threshold)],
        backend=FakeBackend(answer),
    ).evaluate_one(context_sample(tuple(probabilities)))
    metric = result.metrics["context_relevance"]
    assert metric.score == expected
    assert metric.passed is True
    assert metric.details.unit_threshold == unit_threshold


def test_units_have_separate_questions_and_source_context_for_reference_resolution():
    backend = FakeBackend(unit_answers)
    sample = context_sample(("Useful answer. It has a condition.", "Noise passage."))
    Evaluator([ContextRelevance(questions_per_call=2)], backend=backend).evaluate_one(sample)

    assert [len(questions) for _, questions in backend.calls] == [2, 1]
    state, questions = backend.calls[0]
    assert state == {
        "input": sample.input,
        "contexts": ["Useful answer. ", "It has a condition."],
        "context_passages": {"0": sample.contexts[0]},
    }
    assert list(questions) == ["context_relevance__p0_u0", "context_relevance__p0_u1"]
    assert [q.instructions["target_context_index"] for q in questions.values()] == [0, 1]
    assert all(q.instructions["target_passage_index"] == "0" for q in questions.values())
    assert all(
        q.instructions["allowed_evidence_fields"] == ["input", "contexts", "context_passages"]
        for q in questions.values()
    )
    assert sample.contexts[0] not in str([q.instructions for q in questions.values()])
    state, questions = backend.calls[1]
    assert state["context_passages"] == {"1": sample.contexts[1]}
    question = next(iter(questions.values()))
    assert question.instructions["target_context_index"] == 0
    assert question.instructions["target_passage_index"] == "1"


def test_aggregate_reuses_one_compiled_question_template_without_mutating_it():
    compiled = []

    class StatefulMetric(Metric):
        def question(self):
            question = super().question()
            question.instructions["compilation"] = {"ordinal": len(compiled) + 1}
            compiled.append((question, question.model_dump(mode="json")))
            return question

    metric = StatefulMetric(**ContextRelevance(questions_per_call=1).model_dump())
    backend = FakeBackend(unit_answers)
    samples = [context_sample(("Useful evidence.", "Noise passage.")) for _ in range(3)]
    report = Evaluator([metric], backend=backend).evaluate(samples)

    assert len(compiled) == 1
    template, snapshot = compiled[0]
    assert template.model_dump(mode="json") == snapshot
    assert "target_context_index" not in template.instructions
    assert "target_passage_index" not in template.instructions
    questions = [question for _, chunk in backend.calls for question in chunk.values()]
    assert len(questions) == 6
    assert all(question.instructions["compilation"] == {"ordinal": 1} for question in questions)
    assert all(question is not template for question in questions)
    questions[0].instructions["compilation"]["ordinal"] = 999
    assert template.model_dump(mode="json") == snapshot
    assert all(question.instructions["compilation"] == {"ordinal": 1} for question in questions[1:])
    assert all(row.metrics["context_relevance"].score == 0.5 for row in report.results)


def test_aggregate_units_are_physically_separated_from_other_metric_evidence():
    backend = FakeBackend(unit_answers)
    sample = context_sample(("Useful evidence.", "Noise passage."))
    result = Evaluator([AnswerRelevancy(), ContextRelevance()], backend=backend).evaluate_one(
        sample
    )
    assert set(backend.calls[0][1]) == {"answer_relevancy"}
    assert backend.calls[0][0] == {"input": sample.input, "response": sample.response}
    assert set(backend.calls[1][0]) == {"input", "contexts", "context_passages"}
    assert "response" not in backend.calls[1][0]
    assert result.metrics["answer_relevancy"].score == 0.8
    assert result.metrics["context_relevance"].score == 0.5
    assert result.usage == {"input_tokens": 24, "output_tokens": 0}


def test_context_dependent_sibling_sees_complete_original_passages():
    backend = FakeBackend(unit_answers)
    sample = context_sample(("Useful evidence. Noise follows.",))
    Evaluator([Faithfulness(), ContextRelevance()], backend=backend).evaluate_one(sample)
    assert set(backend.calls[0][1]) == {"faithfulness"}
    assert backend.calls[0][0]["contexts"] == list(sample.contexts)
    assert "context_passages" not in backend.calls[0][0]


def test_known_empty_context_is_zero_without_provider_calls():
    backend = FakeBackend(unit_answers)
    result = Evaluator([ContextRelevance()], backend=backend).evaluate_one(context_sample(()))
    metric = result.metrics["context_relevance"]
    assert metric.status == "ok"
    assert metric.raw_kind == "relevance_fraction"
    assert metric.score == 0
    assert metric.passed is False
    assert metric.details.total_units == metric.details.evaluated_units == 0
    assert metric.details.relevant_units == metric.details.unavailable_units == 0
    assert metric.details.units == ()
    assert metric.details.coverage is None
    assert result.model is None
    assert result.usage == {}
    assert backend.calls == []
    assert backend.sessions == backend.closed == 0


def test_known_empty_context_still_evaluates_ordinary_siblings():
    backend = FakeBackend(unit_answers)
    result = Evaluator([ContextRelevance(), AnswerRelevancy()], backend=backend).evaluate_one(
        context_sample(())
    )
    assert result.metrics["context_relevance"].score == 0
    assert result.metrics["answer_relevancy"].score == 0.8
    assert len(backend.calls) == 1


def test_missing_context_preflights_entire_batch_or_skips_without_inventing_zero():
    backend = FakeBackend(unit_answers)
    samples = [context_sample(("Useful evidence.",)), context_sample(None)]
    with pytest.raises(MissingInputError):
        Evaluator([ContextRelevance()], backend=backend).evaluate(samples)
    assert backend.calls == []
    assert backend.sessions == 0
    report = Evaluator([ContextRelevance()], backend=backend, missing="skip").evaluate(samples)
    metric = report.results[1].metrics["context_relevance"]
    assert metric.status == "skipped"
    assert metric.score is metric.passed is metric.details is None
    assert report.summary["context_relevance"].evaluated == 1
    assert report.summary["context_relevance"].skipped == 1


class UnavailableUnitBackend(FakeBackend):
    def __init__(self, failure):
        super().__init__(unit_answers)
        self.failure = failure

    async def judge(self, state, questions):
        response = await super().judge(state, questions)
        target = next((name for name in questions if name.endswith("__p1_u0")), None)
        if target is None:
            return response
        answers = dict(response.answers)
        if self.failure == "transport":
            raise RuntimeError("private context and API credential")
        if self.failure == "invalid":
            answers[target] = {"type": "noul", "noul": float("nan")}
        else:
            answers.pop(target)
        return response.model_copy(
            update={
                "answers": answers,
                "answer_errors": {target: "refusal"} if self.failure == "refusal" else {},
            }
        )


@pytest.mark.parametrize("failure", ["omitted", "invalid", "refusal", "transport"])
def test_unavailable_unit_invalidates_aggregate_preserving_diagnostics_and_siblings(failure):
    backend = UnavailableUnitBackend(failure)
    report = Evaluator(
        [ContextRelevance(questions_per_call=1), AnswerRelevancy()],
        backend=backend,
        errors="record",
    ).evaluate([context_sample(("Useful first.", "Noise second.", "Useful third."))])
    result = report.results[0]
    metric = result.metrics["context_relevance"]
    assert result.metrics["answer_relevancy"].score == 0.8
    assert metric.status == "error"
    assert metric.raw_score is metric.score is metric.passed is None
    assert metric.details.total_units == 3
    assert metric.details.evaluated_units == metric.details.relevant_units == 2
    assert metric.details.unavailable_units == 1
    assert metric.details.coverage == pytest.approx(2 / 3)
    assert [unit.status for unit in metric.details.units] == ["ok", "error", "ok"]
    assert metric.details.units[1].probability is metric.details.units[1].relevant is None
    assert (
        metric.details.units[1].error_type
        == {
            "omitted": "InvalidAnswerError",
            "invalid": "InvalidAnswerError",
            "refusal": "DecisionRefusalError",
            "transport": "RuntimeError",
        }[failure]
    )
    assert len(backend.calls) == 4
    assert "private context" not in str(report.to_dict())
    assert report.summary["context_relevance"].errors == 1
    assert report.summary["context_relevance"].mean_score is None
    assert result.passed is None


@pytest.mark.parametrize(
    "failure,error",
    [
        ("omitted", InvalidAnswerError),
        ("invalid", InvalidAnswerError),
        ("refusal", DecisionRefusalError),
        ("transport", RuntimeError),
    ],
)
def test_unavailable_units_raise_by_default_and_close_session(failure, error):
    backend = UnavailableUnitBackend(failure)
    with pytest.raises(error):
        Evaluator([ContextRelevance()], backend=backend).evaluate_one(
            context_sample(("Useful first.", "Noise second."))
        )
    assert backend.sessions == backend.closed == 1


def test_failed_ordinary_request_does_not_hide_successful_context_aggregate():
    def answer(state, questions):
        if "answer_relevancy" in questions:
            raise RuntimeError("ordinary provider failure")
        return unit_answers(state, questions)

    result = Evaluator(
        [ContextRelevance(), AnswerRelevancy()], backend=FakeBackend(answer), errors="record"
    ).evaluate_one(context_sample(("Useful evidence.",)))
    assert result.metrics["answer_relevancy"].status == "error"
    assert result.metrics["context_relevance"].score == 1


async def test_unit_chunking_preserves_all_units_with_globally_bounded_concurrency():
    backend = FakeBackend(unit_answers, delay=0.001)
    samples = [
        context_sample(tuple("Useful evidence." for _ in range(11))).model_copy(
            update={"id": str(i)}
        )
        for i in range(7)
    ]
    report = await Evaluator(
        [ContextRelevance(questions_per_call=3), AnswerRelevancy()],
        backend=backend,
        max_concurrency=2,
    ).aevaluate(samples)
    assert [result.sample_id for result in report.results] == [str(i) for i in range(7)]
    assert all(
        result.metrics["context_relevance"].details.total_units == 11 for result in report.results
    )
    assert all(result.metrics["context_relevance"].score == 1 for result in report.results)
    assert all(len(questions) <= 3 for _, questions in backend.calls)
    assert backend.peak == 2
    assert backend.active == 0
    assert len(backend.calls) == 7 * 5
    assert backend.sessions == backend.closed == 1
    assert all(
        result.usage == {"input_tokens": 60, "output_tokens": 0} for result in report.results
    )


@pytest.mark.parametrize("errors", ["raise", "record"])
def test_model_drift_between_unit_requests_never_yields_an_aggregate_score(errors):
    class DriftingBackend(FakeBackend):
        async def judge(self, state, questions):
            response = await super().judge(state, questions)
            return response.model_copy(
                update={"model": self.model if len(self.calls) < 3 else "other-model"}
            )

    backend = DriftingBackend(unit_answers)
    evaluator = Evaluator(
        [AnswerRelevancy(), ContextRelevance(questions_per_call=1)],
        backend=backend,
        errors=errors,
    )
    sample = context_sample(("Useful first.", "Useful second."))
    if errors == "raise":
        with pytest.raises(InvalidAnswerError, match="different model"):
            evaluator.evaluate_one(sample)
    else:
        result = evaluator.evaluate_one(sample)
        assert result.model == backend.model
        assert result.metrics["answer_relevancy"].score == 0.8
        assert result.metrics["context_relevance"].status == "error"
        assert result.metrics["context_relevance"].score is None
    assert backend.closed == 1


def test_usage_sums_known_values_and_preserves_unknown_totals():
    class UsageBackend(FakeBackend):
        async def judge(self, state, questions):
            response = await super().judge(state, questions)
            usages = [
                {"input_tokens": 3, "output_tokens": None},
                {"input_tokens": 4, "output_tokens": 0, "cached_tokens": 2},
            ]
            return response.model_copy(update={"usage": usages[len(self.calls) - 1]})

    result = Evaluator(
        [ContextRelevance(questions_per_call=1)], backend=UsageBackend(unit_answers)
    ).evaluate_one(context_sample(("Useful first.", "Useful second.")))
    assert result.usage == {"input_tokens": 7, "output_tokens": None, "cached_tokens": None}


def test_actual_split_requests_are_preflighted_before_session_opens():
    class LimitedBackend(FakeBackend):
        def __init__(self):
            super().__init__(unit_answers)
            self.preflight = []

        def validate_state(self, state):
            assert self.sessions == 0
            self.preflight.append(state)
            assert len(state.get("contexts", ())) <= 2

    backend = LimitedBackend()
    result = Evaluator(
        [ContextRelevance(questions_per_call=2), AnswerRelevancy()], backend=backend
    ).evaluate_one(context_sample(tuple("Useful evidence." for _ in range(5))))
    assert result.metrics["context_relevance"].score == 1
    assert len(backend.preflight) == 4
    assert [len(state.get("contexts", ())) for state in backend.preflight] == [2, 2, 1, 0]


async def test_failure_cancels_other_context_workers_and_closes_shared_session():
    class FailingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = False

        async def judge(self, state, questions):
            if state["input"] == "fail":
                await asyncio.wait_for(self.started.wait(), 5)
                raise RuntimeError("provider failure")
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    backend = FailingBackend()
    samples = [
        EvaluationSample(input=value, response="", contexts=("Useful evidence.",))
        for value in ("wait", "fail")
    ]
    with pytest.raises(RuntimeError, match="provider failure"):
        await Evaluator([ContextRelevance()], backend=backend, max_concurrency=2).aevaluate(samples)
    assert backend.cancelled
    assert backend.sessions == backend.closed == 1
