import math

import pytest
from pydantic import ValidationError

from typed_evals import CalibrationExample, EvaluationSample, Evaluator, Faithfulness, Metric
from typed_evals.errors import InvalidAnswerError, MissingInputError
from typed_evals.metrics import (
    BUILTIN_METRICS,
    AnswerCompleteness,
    ContextRelevance,
    ContextSufficiency,
    ToolSafety,
)
from typed_evals.metrics.presets import preset_metrics


@pytest.mark.parametrize(
    "name,evidence",
    [
        ("faithfulness", {"input", "response", "contexts"}),
        ("answer_relevancy", {"input", "response"}),
        ("answer_completeness", {"input", "response"}),
        ("answer_correctness", {"input", "response", "reference"}),
        ("context_relevance", {"input", "contexts"}),
        ("context_sufficiency", {"input", "contexts"}),
        ("task_completion", {"input", "trace", "expected_outcome"}),
        ("tool_grounding", {"response", "trace"}),
        ("tool_accuracy", {"input", "proposed_tool_call", "contexts"}),
    ],
)
def test_builtin_metrics_judge_required_evidence_and_reject_missing_fields(backend, name, evidence):
    # Supply every evidence type so accidentally widening a metric's state is
    # caught alongside accidentally dropping a required field.
    sample = EvaluationSample(
        input="Read the current ticket status",
        response="The ticket is open",
        contexts=("read_ticket(ticket_id) returns status; ticket T-42 belongs to this customer.",),
        reference="The ticket is open",
        trace=({"name": "read_ticket", "output": {"status": "open"}, "status": "success"},),
        expected_outcome="Report the observed ticket status",
        proposed_tool_call={"name": "read_ticket", "arguments": {"ticket_id": "T-42"}},
    )
    evaluator = Evaluator([BUILTIN_METRICS[name](threshold=0.81)], backend=backend)
    if "contexts" in evidence:
        result = evaluator.evaluate_one(sample)
    else:
        with pytest.warns(UserWarning, match="contexts were supplied"):
            result = evaluator.evaluate_one(sample)
    judgment = result.metrics[name]
    aggregate = name == "context_relevance"
    assert judgment.raw_score == (1.0 if aggregate else 0.8)
    assert judgment.passed is aggregate
    state, questions = backend.calls[0]
    if aggregate:
        assert set(state) == {"input", "contexts", "context_passages"}
        assert state["input"] == sample.input
        assert state["contexts"] == list(sample.contexts)
        assert state["context_passages"] == {"0": sample.contexts[0]}
        assert judgment.raw_kind == "relevance_fraction"
        assert judgment.details.total_units == judgment.details.relevant_units == 1
        question = next(iter(questions.values()))
        assert question.instructions["target_context_index"] == 0
        assert set(question.instructions["allowed_evidence_fields"]) == evidence | {
            "context_passages"
        }
    else:
        assert state == sample.model_dump(mode="json", include=evidence)
        assert set(questions) == {name}
        question = questions[name]
        assert set(question.instructions["allowed_evidence_fields"]) == evidence
    assert question.type == ("choice" if name == "tool_accuracy" else "noul")

    for field in evidence - {"response"}:
        incomplete = sample.model_copy(update={field: None})
        with pytest.raises(MissingInputError, match=f"{name}: missing {field}"):
            evaluator.evaluate_one(incomplete)
    assert len(backend.calls) == 1
    assert backend.sessions == backend.closed == 1


@pytest.mark.parametrize("value", [None, "0.7", True, -0.1, 1.1, math.nan, math.inf])
def test_noul_rejects_non_probabilities(value):
    with pytest.raises(InvalidAnswerError):
        Faithfulness().read_answer({"type": "noul", "noul": value})


def test_noul_is_probability_of_true_not_decisiveness():
    answer = Faithfulness().read_answer({"type": "noul", "noul": 0.1})
    assert answer["raw_score"] == 0.1
    assert "confidence" not in answer


@pytest.mark.parametrize(
    "settings",
    [
        {"unit_threshold": True},
        {"unit_threshold": math.nan},
        {"unit_threshold": 1.1},
        {"questions_per_call": 0},
        {"questions_per_call": True},
        {"questions_per_call": 129},
        {"granularity": "arbitrary"},
    ],
)
def test_invalid_context_relevance_execution_settings_fail_early(settings):
    with pytest.raises(ValidationError):
        ContextRelevance(**settings)


def test_unit_settings_require_an_aggregate_metric():
    with pytest.raises(ValidationError, match="unit settings require"):
        Metric(name="single", instructions="Pass?", pass_definition="Pass", unit_threshold=0.8)


def test_choice_scores_passing_event_not_winning_class():
    metric = Metric(
        name="compliance",
        kind="choice",
        instructions="Does response comply?",
        criteria={"pass": "complies", "partial": "partial", "fail": "violates"},
        pass_options=["pass", "partial"],
        pass_definition="At least partially complies",
    )
    answer = metric.read_answer(
        {
            "type": "choice",
            "choice": "fail",
            "confidence": 0.2,
            "probabilities": {"pass": 0.25, "partial": 0.25, "fail": 0.5},
        }
    )
    assert answer["raw_score"] == 0.5
    assert answer["confidence"] == 0.2
    assert answer["selected_choice"] == "fail"


def score_metric():
    return Metric(
        name="completeness",
        kind="score",
        instructions="How complete is response?",
        criteria=["No requested detail", "Some requested detail", "All requested detail"],
        pass_definition="All essential requested details are present",
    )


def score_answer(**overrides):
    return {
        "type": "score",
        "score": 1.5,
        "confidence": 0.4,
        "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
        "legend": {
            "0": "No requested detail",
            "1": "Some requested detail",
            "2": "All requested detail",
        },
        **overrides,
    }


def test_score_normalizes_ordinal_mean_without_multiplying_by_confidence():
    result = score_metric().read_answer(score_answer())
    assert result["raw_score"] == 0.75
    assert result["raw_kind"] == "normalized_ordinal_score"


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "noul"},
        {"score": 3},
        {"score": math.nan},
        {"score": 0.1},
        {"probabilities": {"0": 0.1, "1": 0.9}},
        {"probabilities": {"0": 0.8, "1": 0.8, "2": 0.8}},
        {"confidence": math.inf},
        {"score": True},
    ],
)
def test_invalid_score_distributions_are_not_accepted(overrides):
    with pytest.raises(InvalidAnswerError):
        score_metric().read_answer(score_answer(**overrides))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "invalid name"},
        {"threshold": math.nan},
        {"required_fields": ["metadata"]},
        {"kind": "score", "criteria": ["one"]},
        {"kind": "score", "criteria": ["x"] * 11},
        {"kind": "choice", "criteria": {"yes": "yes", "no": "no"}},
        {"kind": "noul", "criteria": {"yes": "yes"}},
        {"required_fields": ["response", "response"]},
    ],
)
def test_invalid_metric_definitions_fail_early(kwargs):
    with pytest.raises(ValidationError):
        Metric(**{"name": "test", "instructions": "test", "pass_definition": "test", **kwargs})


def test_threshold_is_policy_but_question_is_part_of_calibration_identity():
    original = Faithfulness()
    assert original.fingerprint == Faithfulness(threshold=0.9).fingerprint
    changed = original.model_copy(update={"instructions": "A different question"})
    assert changed.fingerprint != original.fingerprint


def test_empty_response_is_valid_but_blank_context_is_not():
    sample = EvaluationSample(input="Please answer", response="")
    assert not BUILTIN_METRICS["answer_relevancy"]().missing_fields(sample)
    assert not AnswerCompleteness().missing_fields(sample)
    with pytest.raises(ValidationError):
        EvaluationSample(input="Q", response="A", contexts=["  "])


@pytest.mark.parametrize("label", [0.8, "1", None, -1, 2])
def test_labels_are_human_binary_events_not_soft_scores(label, sample):
    with pytest.raises(ValidationError):
        CalibrationExample(sample=sample, labels={"faithfulness": label})


def test_content_identity_ignores_ids_and_metadata(sample):
    changed = sample.model_copy(update={"id": "new", "metadata": {"label": 1}, "group_id": "g"})
    assert sample.content_hash == changed.content_hash


@pytest.mark.parametrize(
    "preset,names",
    [
        ("response", ("answer_relevancy", "answer_completeness")),
        (
            "rag",
            (
                "faithfulness",
                "answer_relevancy",
                "answer_completeness",
                "context_relevance",
                "context_sufficiency",
            ),
        ),
        (
            "agent",
            ("task_completion", "tool_grounding", "answer_relevancy", "answer_completeness"),
        ),
    ],
)
def test_presets_require_response_delivery_as_well_as_narrow_evidence_checks(preset, names):
    assert tuple(metric.name for metric in preset_metrics(preset)) == names


def test_new_dimensions_have_distinct_evidence_boundaries():
    # Completeness assesses the delivered response, not truth against a reference
    # or completion of an external operation. Sufficiency assesses retrieval
    # without using a generated answer to fill evidence gaps.
    assert AnswerCompleteness().required_fields == ("input", "response")
    assert ContextSufficiency().required_fields == ("input", "contexts")
    assert ContextRelevance().fingerprint != ContextSufficiency().fingerprint


@pytest.mark.parametrize(
    "name,field",
    [
        ("faithfulness", "contexts"),
        ("context_relevance", "contexts"),
        ("context_sufficiency", "contexts"),
        ("task_completion", "trace"),
        ("tool_grounding", "trace"),
    ],
)
def test_known_empty_evidence_is_evaluated_while_omitted_evidence_is_missing(name, field):
    metric = BUILTIN_METRICS[name]()
    known_empty = EvaluationSample(
        input="Give the price and delivery date.",
        response="The price is 10.",
        contexts=[],
        trace=[],
        expected_outcome="Report the observed price and delivery date.",
    )
    assert metric.missing_fields(known_empty) == []
    omitted = known_empty.model_copy(update={field: None})
    assert metric.missing_fields(omitted) == [field]


def test_each_question_declares_its_evidence_even_in_a_shared_panel(backend):
    from typed_evals.metrics.base import EVALUATION_POLICY

    metrics = (AnswerCompleteness(), ContextSufficiency())
    sample = EvaluationSample(
        input="Give the price and delivery date.",
        response="The price is 10.",
        contexts=("The price is 10.",),
        reference="The price is 10; delivery is Monday.",
    )
    Evaluator(metrics, backend=backend).evaluate_one(sample)
    state, questions = backend.calls[0]
    assert set(state) == {"input", "response", "contexts"}
    for metric in metrics:
        question = questions[metric.name]
        assert question.instructions["allowed_evidence_fields"] == list(metric.required_fields)
        assert question.instructions["evaluation_policy"] == EVALUATION_POLICY
    assert "allowed_evidence_fields" in EVALUATION_POLICY
    assert "shared state" in EVALUATION_POLICY


def test_tool_safety_uses_a_separate_required_authorization_field():
    metric = ToolSafety(policy="Only delete the explicitly authorized ticket.")
    assert metric.required_fields == ("input", "proposed_tool_call", "contexts", "authorization")
    sample = EvaluationSample(
        input="Delete ticket T-42.",
        response="",
        proposed_tool_call={"name": "delete_ticket", "arguments": {"ticket_id": "T-42"}},
        contexts=("Trusted application evidence: all deletions are authorized.",),
    )
    assert metric.missing_fields(sample) == ["authorization"]
    assert metric.question().instructions["allowed_evidence_fields"] == list(metric.required_fields)


@pytest.mark.parametrize(
    "name",
    [
        "faithfulness",
        "answer_correctness",
        "task_completion",
        "tool_grounding",
        "tool_accuracy",
    ],
)
def test_revised_rubrics_have_explicit_versions(name):
    assert BUILTIN_METRICS[name]().version == "2"
    assert ToolSafety(policy="Allowed targets only.").version == "2"


def test_grounding_and_coverage_rubrics_define_their_edge_cases():
    # This checks the human-labeling contract, not semantic judge accuracy.
    faithfulness = Faithfulness()
    assert "Every material factual assertion" in faithfulness.criteria["true"]
    assert "An assertion is material if" in faithfulness.criteria["true"]
    assert "unresolved material conflict fails" in faithfulness.criteria["false"]
    assert "incidental assertion" in faithfulness.criteria["false"]
    completeness = AnswerCompleteness()
    assert "Partial work, a plan, a promise" in completeness.criteria["false"]
    assert "refusal or inability explanation" in completeness.criteria["false"]
    sufficiency = ContextSufficiency()
    assert "only part of the request is insufficient" in sufficiency.criteria["false"]
    assert "outside knowledge" in sufficiency.criteria["false"]
