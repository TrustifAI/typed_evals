"""Per-question unavailability preserves siblings without yielding a passing sample."""

import pytest
from conftest import FakeBackend, calibrated_rows

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    JudgeResponse,
    Metric,
)
from typed_evals.errors import DecisionRefusalError, InvalidAnswerError


class PartialBackend(FakeBackend):
    def __init__(self, error_kind):
        super().__init__()
        self.error_kind = error_kind

    async def judge(self, state, questions):
        return JudgeResponse(
            model=self.model,
            answers={"available": {"type": "noul", "noul": 0.9}},
            answer_errors={"unavailable": self.error_kind},
        )


def panel():
    return [
        Metric(name=name, instructions="Does the response pass?", pass_definition="Pass")
        for name in ("available", "unavailable")
    ]


@pytest.mark.parametrize("error_kind", ["refusal", "invalid_answer"])
async def test_partial_answer_record_preserves_success_and_unavailable_pass(error_kind):
    backend = PartialBackend(error_kind)
    report = await Evaluator(panel(), backend=backend, errors="record").aevaluate(
        [EvaluationSample(input="Question", response="Answer")]
    )
    sample = report.results[0]
    assert sample.metrics["available"].raw_score == 0.9
    assert sample.metrics["available"].passed is True
    unavailable = sample.metrics["unavailable"]
    assert unavailable.status == "error"
    assert unavailable.raw_score is unavailable.score is unavailable.passed is None
    assert sample.passed is None
    assert report.summary["available"].evaluated == 1
    assert report.summary["unavailable"].errors == 1
    assert backend.closed == 1


@pytest.mark.parametrize(
    "error_kind,exception",
    [("refusal", DecisionRefusalError), ("invalid_answer", InvalidAnswerError)],
)
async def test_partial_answer_raise_policy_and_session_cleanup(error_kind, exception):
    backend = PartialBackend(error_kind)
    with pytest.raises(exception):
        await Evaluator(panel(), backend=backend).aevaluate(
            [EvaluationSample(input="Question", response="Answer")]
        )
    assert backend.closed == 1


class RefitBackend(FakeBackend):
    unavailable = False

    async def judge(self, state, questions):
        if self.unavailable:
            return JudgeResponse(
                model=self.model,
                answers={},
                answer_errors={name: "refusal" for name in questions},
            )
        return JudgeResponse(
            model=self.model,
            answers={
                name: {"type": "noul", "noul": float(state["response"])} for name in questions
            },
        )


def test_failed_refit_with_unavailable_labeled_judgment_preserves_valid_bundle():
    backend = RefitBackend()
    pipeline = EvaluationPipeline(
        [AnswerRelevancy()],
        backend=backend,
        calibration=CalibrationConfig(enabled=True, min_samples=20, min_validation_samples=10),
        errors="record",
    )
    rows, validation = calibrated_rows(), calibrated_rows("validation")
    pipeline.fit(rows, validation_data=validation)
    previous = pipeline.calibration_bundle
    backend.unavailable = True
    with pytest.raises(DecisionRefusalError):
        pipeline.fit(rows, validation_data=validation)
    assert pipeline.calibration_bundle is previous
    assert not pipeline._fitting
