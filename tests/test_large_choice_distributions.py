"""High-cardinality Choice answers retain bounded mass validation across backends."""

import pytest
from conftest import FakeBackend

from typed_evals import EvaluationSample, Evaluator, Metric
from typed_evals.backends.openai_decisions import _distribution, compile_questions
from typed_evals.errors import InvalidAnswerError


def choice_metric(option_count):
    return Metric(
        name="many_options",
        kind="choice",
        instructions="Choose the matching option.",
        criteria={str(index): f"Option {index}" for index in range(option_count)},
        pass_options=("0",),
        pass_definition="Option zero matches.",
    )


def choice_answer(option_count, probability):
    return {
        "type": "choice",
        "choice": "0",
        "confidence": 0.5,
        "probabilities": {str(index): probability for index in range(option_count)},
    }


def decisions_distribution(metric, answer):
    question = compile_questions({metric.name: metric.question()})[0]
    return _distribution(
        {
            "probabilities": [
                {"value": key, "probability": value}
                for key, value in answer["probabilities"].items()
            ]
        },
        question,
    )


@pytest.mark.parametrize("option_count", [200, 255])
@pytest.mark.parametrize("total", [0, 0.75, 0.95, 1.05, 1.25])
def test_large_choice_rejects_materially_invalid_mass(option_count, total):
    metric = choice_metric(option_count)
    answer = choice_answer(option_count, total / option_count)
    with pytest.raises(InvalidAnswerError, match="sum approximately to one"):
        metric.read_answer(answer)
    # The shared Jev/custom-backend parser follows the same bound as Native Decisions.
    with pytest.raises(InvalidAnswerError, match="sum approximately to one"):
        decisions_distribution(metric, answer)


@pytest.mark.parametrize("option_count", [200, 255])
def test_large_choice_accepts_valid_distribution(option_count):
    metric = choice_metric(option_count)
    answer = choice_answer(option_count, 1 / option_count)
    result = metric.read_answer(answer)
    assert result["raw_score"] == pytest.approx(1 / option_count)
    assert result["selected_choice"] == "0"
    assert result["confidence"] == 0.5
    assert result["probabilities"] == decisions_distribution(metric, answer)


@pytest.mark.parametrize("option_count", [200, 255])
async def test_large_choice_scores_passing_mass_even_when_the_selected_option_fails(option_count):
    metric = choice_metric(option_count).model_copy(
        update={"pass_options": ("2", "10"), "threshold": 0.5}
    )
    probabilities = dict.fromkeys(metric.criteria, 0.0)
    probabilities.update({"0": 0.45, "2": 0.30, "10": 0.25})
    answer = {
        "type": "choice",
        "choice": "0",
        "confidence": 0.95,
        "probabilities": probabilities,
    }
    backend = FakeBackend(lambda state, questions: {metric.name: answer})

    report = await Evaluator([metric], backend=backend).aevaluate(
        [EvaluationSample(input="Choose the category", response="candidate")]
    )

    result = report.results[0].metrics[metric.name]
    assert result.raw_score == pytest.approx(0.55)
    assert result.score == pytest.approx(0.55)
    assert result.selected_choice == "0"
    assert result.confidence == 0.95
    assert result.probabilities == probabilities
    assert result.passed is report.results[0].passed is True
    assert report.summary[metric.name].mean_raw_score == pytest.approx(0.55)
    assert report.summary[metric.name].errors == 0
    assert decisions_distribution(metric, answer) == probabilities
    assert len(backend.calls) == 1
    assert len(backend.calls[0][1][metric.name].criteria) == option_count


@pytest.mark.parametrize(
    "option_count,probability",
    [(3, 0.33), (200, 0.0049), (200, 0.0051), (255, 0.00392), (255, 0.004)],
)
def test_choice_accepts_supported_rounding(option_count, probability):
    metric = choice_metric(option_count)
    answer = choice_answer(option_count, probability)
    result = metric.read_answer(answer)
    assert result["raw_score"] == probability
    assert result["probabilities"] == decisions_distribution(metric, answer)


@pytest.mark.parametrize("option_count", [200, 255])
@pytest.mark.parametrize("total", [0, 0.75, 1.25])
@pytest.mark.parametrize("invalid_first", [False, True])
async def test_invalid_large_choice_records_unavailable_and_preserves_sibling(
    option_count, total, invalid_first
):
    invalid_metric = choice_metric(option_count)
    sibling = Metric(
        name="available",
        instructions="Does the response pass?",
        pass_definition="The response passes.",
    )
    metrics = [invalid_metric, sibling] if invalid_first else [sibling, invalid_metric]
    backend = FakeBackend(
        lambda state, questions: {
            invalid_metric.name: choice_answer(option_count, total / option_count),
            sibling.name: {"type": "noul", "noul": 0.9},
        }
    )
    report = await Evaluator(metrics, backend=backend, errors="record").aevaluate(
        [EvaluationSample(input="Question", response="Answer")]
    )
    row = report.results[0]
    unavailable = row.metrics[invalid_metric.name]
    assert unavailable.status == "error"
    assert unavailable.raw_score is unavailable.score is unavailable.passed is None
    assert "InvalidAnswerError" in unavailable.message
    assert row.metrics[sibling.name].raw_score == 0.9
    assert row.metrics[sibling.name].passed is True
    assert row.passed is None
    assert report.summary[invalid_metric.name].evaluated == 0
    assert report.summary[invalid_metric.name].errors == 1
    assert report.summary[sibling.name].evaluated == 1
    assert backend.sessions == backend.closed == 1
