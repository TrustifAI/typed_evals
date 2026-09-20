from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from typed_evals import CalibrationExample, EvaluationSample, JudgeResponse


def binary_answer(question, score):
    if question.type == "choice":
        return {
            "type": "choice",
            "choice": "true" if score >= 0.5 else "false",
            "confidence": 0.6,
            "probabilities": {"true": score, "false": 1 - score},
        }
    return {"type": "noul", "noul": score}


class FakeBackend:
    model = "test-jev-1"

    def __init__(self, answer=None, delay=0):
        self.answer = answer
        self.delay = delay
        self.calls = []
        self.active = 0
        self.peak = 0
        self.sessions = 0
        self.closed = 0
        self.actual_model = self.model

    @asynccontextmanager
    async def session(self):
        self.sessions += 1
        try:
            yield self
        finally:
            self.closed += 1

    async def judge(self, state, questions):
        self.calls.append((state, questions))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.answer is not None:
                answers = self.answer(state, questions)
            else:
                answers = {
                    name: binary_answer(question, 0.8) for name, question in questions.items()
                }
            return JudgeResponse(
                model=self.actual_model,
                answers=answers,
                usage={"input_tokens": 12, "output_tokens": 0},
            )
        finally:
            self.active -= 1


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def sample():
    return EvaluationSample(
        input="What is the capital of France?",
        response="Paris.",
        contexts=["The capital of France is Paris."],
        reference="Paris.",
        id="france",
    )


def calibrated_rows(prefix="train", size=40, group_id=None):
    """Each score group has a known event rate; independent train/validation IDs."""
    rows = []
    for index in range(size):
        bucket = index % 2
        # A raw 0.2 group has 40% positives; raw 0.8 group has 60% positives.
        label = int((index // 2) % 5 < (2 if bucket == 0 else 3))
        rows.append(
            CalibrationExample(
                sample=EvaluationSample(
                    input=f"{prefix}-{index}",
                    response=str(0.2 if bucket == 0 else 0.8),
                    group_id=group_id,
                ),
                labels={"answer_relevancy": label},
            )
        )
    return rows


@pytest.fixture
def scoring_backend():
    return FakeBackend(
        lambda state, questions: {
            name: {"type": "noul", "noul": float(state["response"])} for name in questions
        }
    )
