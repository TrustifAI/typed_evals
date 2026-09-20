import os

import pytest

from typed_evals import AnswerCorrectness, Evaluator, Faithfulness, JevBackend


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
