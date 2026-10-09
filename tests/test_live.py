import os

import pytest

from typed_evals import (
    AnswerCorrectness,
    Evaluator,
    Faithfulness,
    JevBackend,
    Metric,
    OpenAIDecisionsBackend,
)


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("TYPED_EVALS_LIVE") != "1" or not os.getenv("TYPESAFE_API_KEY"),
    reason="Set TYPED_EVALS_LIVE=1 and TYPESAFE_API_KEY to run a billable API smoke test",
)
def test_live_jev_smoke(sample):
    backend = JevBackend(model=os.getenv("TYPED_EVALS_MODEL", "jev-1.13.0"))
    result = Evaluator([Faithfulness(), AnswerCorrectness()], backend=backend).evaluate_one(sample)
    assert result.model
    # Smoke-test the live contract, not a universal accuracy claim from one example.
    assert all(
        value.status == "ok" and 0 <= value.raw_score <= 1 for value in result.metrics.values()
    )


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("TYPED_EVALS_OPENAI_LIVE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set TYPED_EVALS_OPENAI_LIVE=1 and OPENAI_API_KEY to run a billable Decisions smoke test",
)
def test_live_openai_decisions_smoke(sample):
    pytest.importorskip("openai", minversion="3.26.0")
    metrics = [
        Faithfulness(),
        Metric(
            name="answer_verdict",
            kind="choice",
            instructions="Does response correctly answer input using the supplied contexts?",
            criteria={"true": "The answer is supported and correct.", "false": "It is incorrect."},
            pass_options=("true",),
            pass_definition="The answer is supported and correct.",
        ),
        Metric(
            name="coverage",
            kind="score",
            instructions="How much of the information requested in input is supplied by response?",
            criteria=("None of it.", "Some of it.", "All of it."),
            pass_definition="All requested information is supplied.",
        ),
    ]
    result = Evaluator(
        metrics,
        backend=OpenAIDecisionsBackend(model=os.getenv("TYPED_EVALS_OPENAI_MODEL", "gpt-6-luna")),
    ).evaluate_one(sample)
    assert result.model
    assert set(result.metrics) == {metric.name for metric in metrics}
    # Validate the live wire contract, without asserting accuracy from a single row.
    assert all(
        value.status == "ok" and 0 <= value.raw_score <= 1 for value in result.metrics.values()
    )
    assert result.metrics["faithfulness"].confidence is None
