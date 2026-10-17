"""Aggregate retrieval judgments round-trip through every bundled wire protocol."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from typed_evals import (
    AnswerRelevancy,
    ContextRelevance,
    EvaluationSample,
    Evaluator,
    JevBackend,
    OpenAIDecisionsBackend,
    SystemOneBackend,
)

MODELS = {
    "jev": "jev-1.14.0",
    "openai": "gpt-6-luna-observed",
    "systemone": "test-systemone-model",
}


@asynccontextmanager
async def mock_backend(provider, handler):
    if provider == "jev":
        async with AsyncTypeSafeClient(
            api_key="test-only-not-a-real-key",
            transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
            timeout=2,
        ) as client:
            yield JevBackend(client=client)
    elif provider == "openai":
        openai = pytest.importorskip("openai", minversion="3.26.0")
        async with openai.AsyncOpenAI(
            api_key="test-only-not-a-real-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
            max_retries=0,
        ) as client:
            yield OpenAIDecisionsBackend(client=client)
    else:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
            yield SystemOneBackend(
                model=MODELS[provider],
                base_url="https://mock-judge.example.test",
                client=client,
                max_retries=0,
            )


def decode_request(provider, request):
    body = json.loads(request.content)
    if provider == "openai":
        assert request.url.path == "/v1/decisions"
        state = json.loads(body["input"])
        questions = {
            question["name"]: json.loads(question["instructions"].split("\n", 1)[0])
            for question in body["questions"]
        }
        assert all(question["type"] == "predicate" for question in body["questions"])
    else:
        assert request.url.path == "/v1/systemone"
        state = body["state"]
        questions = {name: question["instructions"] for name, question in body["questions"].items()}
        assert all(question["type"] == "noul" for question in body["questions"].values())
    return state, questions


def wire_response(provider, probabilities):
    usage = {"input_tokens": 7, "output_tokens": 0}
    if provider == "openai":
        answers = [
            {"name": name, "type": "predicate", "probability": probability}
            for name, probability in reversed(probabilities.items())
        ]
        usage["total_tokens"] = 7
    else:
        answers = {
            name: {"type": "noul", "noul": probability}
            for name, probability in reversed(probabilities.items())
        }
    return httpx2.Response(
        200, json={"model": MODELS[provider], "answers": answers, "usage": usage}
    )


def scoring_handler(provider, observed, *, omit_unit=False):
    def handler(request):
        state, questions = decode_request(provider, request)
        observed.append((state, questions))
        probabilities = {}
        for name, instructions in questions.items():
            if omit_unit and name == "context_relevance__p1_u0":
                continue
            target = instructions.get("target_context_index")
            probabilities[name] = (
                (0.9 if state["contexts"][target].startswith("Useful") else 0.1)
                if target is not None
                else 0.8
            )
        return wire_response(provider, probabilities)

    return handler


@pytest.mark.parametrize("provider", ["jev", "openai", "systemone"])
@pytest.mark.parametrize("questions_per_call", [32, 1])
async def test_real_backend_serialization_preserves_unit_selection_and_fraction(
    provider, questions_per_call
):
    observed = []
    sample = EvaluationSample(
        input="What is useful?",
        response="Delivered answer.",
        contexts=("Useful evidence.", "Noise passage."),
        reference="private unused reference",
        metadata={"private": "never evidence"},
    )
    async with mock_backend(provider, scoring_handler(provider, observed)) as backend:
        result = await Evaluator(
            [AnswerRelevancy(), ContextRelevance(questions_per_call=questions_per_call)],
            backend=backend,
        ).aevaluate_one(sample)

    assert observed[0] == (
        {"input": sample.input, "response": sample.response},
        {"answer_relevancy": AnswerRelevancy().question().instructions},
    )
    assert len(observed) == (2 if questions_per_call == 32 else 3)
    names, target_indices = [], []
    for state, questions in observed[1:]:
        assert set(state) == {"input", "contexts", "context_passages"}
        assert len(questions) <= questions_per_call
        for name, instructions in questions.items():
            names.append(name)
            target = instructions["target_context_index"]
            passage = instructions["target_passage_index"]
            target_indices.append(target)
            assert instructions["allowed_evidence_fields"] == [
                "input",
                "contexts",
                "context_passages",
            ]
            assert state["contexts"][target] == sample.contexts[int(passage)]
            assert state["context_passages"][passage] == sample.contexts[int(passage)]
            assert sample.contexts[int(passage)] not in json.dumps(instructions)
    assert names == ["context_relevance__p0_u0", "context_relevance__p1_u0"]
    assert target_indices == ([0, 1] if questions_per_call == 32 else [0, 0])
    assert result.model == MODELS[provider]
    assert result.metrics["answer_relevancy"].score == 0.8
    metric = result.metrics["context_relevance"]
    assert metric.raw_kind == "relevance_fraction"
    assert metric.raw_score == metric.score == 0.5
    assert metric.passed is True
    assert metric.details.total_units == metric.details.evaluated_units == 2
    assert metric.details.relevant_units == 1
    assert metric.details.unavailable_units == 0
    assert metric.details.coverage == 1
    assert [unit.probability for unit in metric.details.units] == [0.9, 0.1]
    assert [unit.relevant for unit in metric.details.units] == [True, False]
    assert result.usage["input_tokens"] == 7 * len(observed)
    assert result.usage["output_tokens"] == 0


@pytest.mark.parametrize("provider", ["jev", "openai", "systemone"])
async def test_real_backend_omitted_unit_preserves_sibling_without_partial_fraction(provider):
    observed = []
    sample = EvaluationSample(
        input="What is useful?",
        response="Delivered answer.",
        contexts=("Useful evidence.", "Noise passage."),
    )
    async with mock_backend(
        provider, scoring_handler(provider, observed, omit_unit=True)
    ) as backend:
        result = await Evaluator(
            [AnswerRelevancy(), ContextRelevance()], backend=backend, errors="record"
        ).aevaluate_one(sample)

    assert len(observed) == 2
    assert result.metrics["answer_relevancy"].score == 0.8
    assert result.metrics["answer_relevancy"].status == "ok"
    metric = result.metrics["context_relevance"]
    assert metric.status == "error"
    assert metric.raw_score is metric.score is metric.passed is None
    assert metric.details.total_units == 2
    assert metric.details.unavailable_units >= 1
    assert metric.details.coverage < 1
    assert result.passed is None
