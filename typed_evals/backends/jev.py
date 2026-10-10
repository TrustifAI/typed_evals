"""Official SDK adapter. One session pools connections across a whole batch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from pydantic import Field
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score

from typed_evals.backends.capabilities import validate_evidence
from typed_evals.data.models import Model

Question = Noul | Choice | Score


class JudgeResponse(Model):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int | None] = Field(default_factory=dict)
    # Optional, independently attributable failures; successful sibling answers survive.
    answer_errors: dict[str, Literal["invalid_answer", "refusal"]] = Field(default_factory=dict)


class JudgeSession(Protocol):
    async def judge(
        self, state: dict[str, Any], questions: Mapping[str, Question]
    ) -> JudgeResponse: ...


class Backend(Protocol):
    """Optional supported_modalities and validate_state hooks extend text backends."""

    model: str

    def session(self) -> AbstractAsyncContextManager[JudgeSession]: ...


@dataclass
class _SDKSession:
    client: AsyncTypeSafeClient
    model: str
    supported_modalities: frozenset[str] = frozenset({"text"})

    async def judge(
        self, state: dict[str, Any], questions: Mapping[str, Question]
    ) -> JudgeResponse:
        validate_evidence(self, state)
        result = await self.client.system_one(state=state, questions=questions, model=self.model)
        return JudgeResponse(
            model=result.model,
            answers={key: answer.model_dump(mode="json") for key, answer in result.answers.items()},
            usage=result.usage.model_dump(),
        )


@dataclass
class JevBackend:
    supported_modalities: ClassVar[frozenset[str]] = frozenset({"text"})

    model: str = "jev-1.13.0"
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 30.0
    max_retries: int = 2
    client: AsyncTypeSafeClient | None = field(default=None, repr=False)
    base_url: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import math

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty identifier")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")

    def calibration_provenance(self, questions: Mapping[str, Question]) -> dict[str, Any]:
        """Identify the existing SDK question translation independently of transport."""
        return {
            "provider": "typesafe",
            "backend": "jev",
            "compiler_version": "1",
            "questions": [
                {
                    "name": name,
                    "schema": question.model_dump(mode="json"),
                    # Preserve option ordering even though fingerprinting sorts object keys.
                    "criteria_order": list(question.criteria)
                    if question.criteria is not None
                    else None,
                }
                for name, question in questions.items()
            ],
        }

    @asynccontextmanager
    async def session(self) -> AsyncIterator[JudgeSession]:
        if self.client is not None:
            # Injected clients belong to the caller, including their retry/timeout configuration.
            yield _SDKSession(self.client, self.model, self.supported_modalities)
            return
        async with AsyncTypeSafeClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            retry=RetryPolicy(max_retries=self.max_retries, timeout=self.timeout),
        ) as client:
            yield _SDKSession(client, self.model, self.supported_modalities)
