"""Shared backend contracts and the extension base for provider adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError
from typesafe_sdk import Choice, Noul, Score

from typed_evals.data.models import ImageInput, Model

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
    """Structural contract retained for custom backends that do not inherit the base."""

    model: str

    def session(self) -> AbstractAsyncContextManager[JudgeSession]: ...


class BaseBackend(ABC):
    """Extend this base to implement a new provider's judge transport.

    Dataclass subclasses inherit configuration validation through ``__post_init__``;
    overrides must call ``super().__post_init__()``. Keep client creation and cleanup
    in ``session()``. Shared ``image_inputs`` validation handles neutral evidence;
    override ``supported_modalities`` and ``validate_state`` for provider capabilities
    and input limits. Calibration provenance remains optional.

    This base is not a dataclass, so adapters retain control of their constructor
    fields and ordering, including provider-specific required fields.
    """

    model: str
    timeout: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty identifier")
        try:
            valid_timeout = (
                type(self.timeout) in (int, float)
                and math.isfinite(self.timeout)
                and self.timeout > 0
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("timeout must be positive and finite")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")

    @property
    def supported_modalities(self) -> frozenset[str]:
        """Text is the default; providers explicitly opt into other evidence."""
        return frozenset({"text"})

    @staticmethod
    def image_inputs(state: Mapping[str, Any]) -> tuple[ImageInput, ...]:
        """Validate neutral image evidence without imposing a provider's wire format."""
        images = state.get("images")
        if images is None:
            return ()
        if not isinstance(images, (list, tuple)):
            raise ValueError("images must be a sequence of ImageInput evidence")
        try:
            return tuple(
                ImageInput.model_validate(
                    image.model_dump(warnings=False) if isinstance(image, ImageInput) else image
                )
                for image in images
            )
        except ValidationError:
            # Provider input errors must not echo inline image payloads.
            raise ValueError("images must contain valid ImageInput evidence") from None

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """Override to preflight provider limits before a batch opens its session."""
        self.image_inputs(state)

    @abstractmethod
    def session(self) -> AbstractAsyncContextManager[JudgeSession]:
        """Pool provider resources for a batch and yield a session that judges samples."""
        raise NotImplementedError
