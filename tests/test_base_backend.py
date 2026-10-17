"""A provider can extend the common backend API without a built-in transport."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from conftest import FakeBackend

import typed_evals
import typed_evals.backends as backends
import typed_evals.backends.jev as jev
from typed_evals import (
    AnswerRelevancy,
    BaseBackend,
    EvaluationSample,
    Evaluator,
    ImageInput,
    JevBackend,
    JudgeResponse,
    JudgeSession,
    Metric,
    OpenAIDecisionsBackend,
    SystemOneBackend,
)
from typed_evals.backends import base
from typed_evals.backends.base import Question
from typed_evals.backends.provenance import backend_provenance
from typed_evals.errors import UnsupportedModalityError


@dataclass
class LocalBackend(BaseBackend):
    model: str = "local-judge"
    sessions: int = field(default=0, init=False)
    closed: int = field(default=0, init=False)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[JudgeSession]:
        self.sessions += 1
        try:
            yield self
        finally:
            self.closed += 1

    async def judge(
        self, state: dict[str, Any], questions: Mapping[str, Question]
    ) -> JudgeResponse:
        self.calls.append(state)
        return JudgeResponse(
            model=self.model,
            answers={name: {"type": "noul", "noul": 0.8} for name in questions},
        )


@dataclass
class ConfigurableLocalBackend(LocalBackend):
    timeout: float = 30.0
    max_retries: int = 2


class SingleImageBackend(LocalBackend):
    @property
    def supported_modalities(self) -> frozenset[str]:
        return frozenset({"text", "image"})

    def validate_state(self, state: Mapping[str, Any]) -> None:
        if len(state.get("images", ())) > 1:
            raise ValueError("at most one image is supported")


def image_metric() -> Metric:
    return Metric(
        name="visual_grounding",
        instructions="Is the response supported by the image?",
        required_fields=("input", "response", "images"),
        pass_definition="The response is supported by the attached image.",
    )


def image_sample() -> EvaluationSample:
    return EvaluationSample(
        input="Describe the product.",
        response="The screen is damaged.",
        images=[ImageInput.from_bytes(b"fixture", mime_type="image/png")],
    )


def test_shared_contracts_have_stable_public_and_legacy_imports():
    assert typed_evals.BaseBackend is backends.BaseBackend is base.BaseBackend
    for name in ("Backend", "JudgeResponse", "JudgeSession", "Question"):
        assert getattr(backends, name) is getattr(base, name) is getattr(jev, name)
    for name in ("Backend", "JudgeResponse", "JudgeSession"):
        assert getattr(typed_evals, name) is getattr(base, name)
    for provider in (JevBackend, OpenAIDecisionsBackend, SystemOneBackend):
        assert issubclass(provider, BaseBackend)


def test_backend_requires_a_session_implementation():
    @dataclass
    class MissingSession(BaseBackend):
        model: str = "unfinished-judge"

    with pytest.raises(TypeError, match="abstract.*session"):
        MissingSession()


async def test_minimal_dataclass_backend_uses_shared_defaults_and_evaluator():
    backend = LocalBackend()
    samples = [
        EvaluationSample(input="first question", response="first answer"),
        EvaluationSample(input="second question", response="second answer"),
    ]

    report = await Evaluator([AnswerRelevancy()], backend=backend).aevaluate(samples)

    assert backend.timeout == 30.0
    assert backend.max_retries == 2
    assert backend.supported_modalities == frozenset({"text"})
    assert backend.sessions == backend.closed == 1
    assert backend.calls == [sample.state({"input", "response"}) for sample in samples]
    assert all(row.metrics["answer_relevancy"].score == 0.8 for row in report.results)
    assert all(row.passed for row in report.results)
    assert backend_provenance(backend, {}) == {"status": "unknown"}


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("model", None),
        ("model", True),
        ("model", 42),
        ("model", ""),
        ("model", "  "),
        ("timeout", True),
        ("timeout", False),
        ("timeout", None),
        ("timeout", "30"),
        ("timeout", 0),
        ("timeout", -1),
        ("timeout", float("inf")),
        ("timeout", float("nan")),
        ("timeout", 10**400),
        ("max_retries", True),
        ("max_retries", False),
        ("max_retries", None),
        ("max_retries", "2"),
        ("max_retries", 1.0),
        ("max_retries", 1.5),
        ("max_retries", -1),
    ],
)
def test_custom_dataclass_backend_inherits_configuration_validation(field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        ConfigurableLocalBackend(**{field_name: invalid})


@pytest.mark.parametrize("provider", [JevBackend, OpenAIDecisionsBackend, SystemOneBackend])
@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [("model", " "), ("timeout", True), ("max_retries", True)],
)
def test_provider_backends_apply_shared_configuration_validation(provider, field_name, invalid):
    options = {"model": "judge-model", field_name: invalid}
    if provider is SystemOneBackend:
        options["base_url"] = "https://judge.example.com"
    with pytest.raises(ValueError, match=field_name):
        provider(**options)


async def test_base_backend_rejects_images_before_opening_any_batch_session():
    backend = LocalBackend()
    evaluator = Evaluator([AnswerRelevancy(), image_metric()], backend=backend, missing="skip")
    with pytest.raises(UnsupportedModalityError, match="does not support image evidence"):
        await evaluator.aevaluate(
            [EvaluationSample(input="text first", response="fine"), image_sample()]
        )
    assert backend.sessions == 0
    assert backend.calls == []


async def test_custom_backend_can_opt_into_images_without_provider_transport():
    backend = SingleImageBackend()
    sample = image_sample()

    row = await Evaluator([image_metric()], backend=backend).aevaluate_one(sample)

    assert row.passed
    assert backend.sessions == backend.closed == 1
    assert backend.calls == [sample.state({"input", "response", "images"})]


async def test_custom_provider_validation_preflights_every_row_before_session():
    backend = SingleImageBackend()
    valid = image_sample()
    oversized = valid.model_copy(update={"images": valid.images * 2})

    with pytest.raises(ValueError, match="at most one image"):
        await Evaluator([image_metric()], backend=backend).aevaluate([valid, oversized])

    assert backend.sessions == 0
    assert backend.calls == []


async def test_decision_one_can_use_jev_transport_with_images_explicitly_disabled(monkeypatch):
    backend = JevBackend(
        model="microsoft/microsoft-decision-1",
        base_url="https://openrouter.ai/api",
        image_mode="disabled",
    )

    def unexpected_session():
        pytest.fail("Unsupported images must be rejected before opening the SDK session")

    monkeypatch.setattr(backend, "session", unexpected_session)
    with pytest.raises(UnsupportedModalityError, match="JevBackend"):
        await Evaluator([image_metric()], backend=backend).aevaluate_one(image_sample())


async def test_existing_structural_backends_remain_usable_without_inheriting_base():
    backend = FakeBackend()
    row = await Evaluator(backend=backend).aevaluate_one(
        EvaluationSample(input="question", response="answer")
    )
    assert row.passed
    assert not isinstance(backend, BaseBackend)
    assert backend.sessions == backend.closed == 1
    assert backend_provenance(backend, {}) == {"status": "unknown"}
