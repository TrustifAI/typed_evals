import math

import pytest
from pydantic import ValidationError

from typed_evals import CalibrationExample, EvaluationSample, Faithfulness, Metric
from typed_evals.errors import InvalidAnswerError
from typed_evals.metrics import BUILTIN_METRICS


@pytest.mark.parametrize("name", BUILTIN_METRICS)
def test_builtin_questions_use_real_sdk_types(name):
    metric = BUILTIN_METRICS[name]()
    question = metric.question()
    assert question.type == ("choice" if name == "tool_accuracy" else "noul")
    assert "state" in question.instructions["evaluation_policy"]
    assert question.criteria["true"] and question.criteria["false"]


@pytest.mark.parametrize("value", [None, "0.7", True, -0.1, 1.1, math.nan, math.inf])
def test_noul_rejects_non_probabilities(value):
    with pytest.raises(InvalidAnswerError):
        Faithfulness().read_answer({"type": "noul", "noul": value})


def test_noul_is_probability_of_true_not_decisiveness():
    answer = Faithfulness().read_answer({"type": "noul", "noul": 0.1})
    assert answer["raw_score"] == 0.1
    assert "confidence" not in answer


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
    with pytest.raises(ValidationError):
        EvaluationSample(input="Q", response="A", contexts=["  "])


@pytest.mark.parametrize("label", [0.8, "1", None, -1, 2])
def test_labels_are_human_binary_events_not_soft_scores(label, sample):
    with pytest.raises(ValidationError):
        CalibrationExample(sample=sample, labels={"faithfulness": label})


def test_content_identity_ignores_ids_and_metadata(sample):
    changed = sample.model_copy(update={"id": "new", "metadata": {"label": 1}, "group_id": "g"})
    assert sample.content_hash == changed.content_hash
