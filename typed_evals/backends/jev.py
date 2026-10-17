"""Official SDK adapter. One session pools connections across a whole batch."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from typesafe_sdk import AsyncTypeSafeClient, Image, RetryPolicy
from typesafe_sdk.constants import BASE_URL_ENV, DEFAULT_BASE_URL

from typed_evals.backends.base import Backend as Backend
from typed_evals.backends.base import BaseBackend
from typed_evals.backends.base import JudgeResponse as JudgeResponse
from typed_evals.backends.base import JudgeSession as JudgeSession
from typed_evals.backends.base import Question as Question
from typed_evals.backends.capabilities import validate_evidence
from typed_evals.data.models import ImageInput
from typed_evals.errors import CalibrationError

ImageMode = Literal["disabled", "experimental", "native"]
COMPILER_VERSION = "2"
IMAGE_COMPILER_VERSION = "images-1"


def _modalities(image_mode: ImageMode) -> frozenset[str]:
    return frozenset({"text"}) if image_mode == "disabled" else frozenset({"text", "image"})


def _validated_images(state: Mapping[str, Any]) -> tuple[ImageInput, ...]:
    validated = BaseBackend.image_inputs(state)
    for image in validated:
        if not image.data_url.startswith(("data:image/png;", "data:image/jpeg;")):
            raise ValueError("Jev accepts only PNG or JPEG image evidence")
        if image.detail != "auto":
            raise ValueError("Jev does not support image detail controls; use detail='auto'")
    return validated


def _compile_state(state: dict[str, Any]) -> dict[str, Any]:
    """Delegate asset references and multipart transport to the official SDK."""
    images = _validated_images(state)
    if not images:
        return state
    return {
        **state,
        "images": [
            {"image": Image.from_base64(image.data_url.partition(",")[2]), "label": image.label}
            for image in images
        ],
    }


def _sdk_client_api_root(client: AsyncTypeSafeClient) -> str:
    """Isolate the resolved configuration API used by supported SDK 0.7.x clients."""
    endpoint = getattr(getattr(client, "_config", None), "base_url", None)
    if not isinstance(endpoint, str):
        raise CalibrationError(
            "Jev cannot establish the injected SDK client's endpoint from _config.base_url"
        )
    return endpoint


def _normalize_api_root(endpoint: str) -> str:
    """Keep routing identity while removing URL credentials and cosmetic differences."""
    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname
        port = parsed.port
        if parsed.scheme not in ("https", "http") or not host or any(c.isspace() for c in host):
            raise ValueError
        if ":" in host:
            host = f"[{host}]"
        if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
            host += f":{port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/") + "/", "", ""))
    except (TypeError, ValueError):
        raise CalibrationError("Jev requires a resolved HTTP(S) API root for provenance") from None


@dataclass
class _SDKSession:
    client: AsyncTypeSafeClient
    model: str
    image_mode: ImageMode = "disabled"

    @property
    def supported_modalities(self) -> frozenset[str]:
        return _modalities(self.image_mode)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        _validated_images(state)

    async def judge(
        self, state: dict[str, Any], questions: Mapping[str, Question]
    ) -> JudgeResponse:
        validate_evidence(self, state)
        options: dict[str, Any] = {}
        if state.get("images") and self.image_mode == "experimental":
            options["extra_headers"] = {"TypeSafe-Danger-Zone-Images": "true"}
        result = await self.client.system_one(
            state=_compile_state(state), questions=questions, model=self.model, **options
        )
        return JudgeResponse(
            model=result.model,
            answers={key: answer.model_dump(mode="json") for key, answer in result.answers.items()},
            usage=result.usage.model_dump(),
        )


@dataclass
class JevBackend(BaseBackend):
    """SDK adapter with explicit image opt-in, independent of the model identifier.

    Experimental and native images use the same SDK transport; native mode omits
    the experimental header for endpoints that no longer require it.
    """

    model: str = "jev-1.14.0"
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 30.0
    max_retries: int = 2
    client: AsyncTypeSafeClient | None = field(default=None, repr=False)
    base_url: str | None = field(default=None, repr=False)
    image_mode: ImageMode = "disabled"

    @property
    def supported_modalities(self) -> frozenset[str]:
        return _modalities(self.image_mode)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        _validated_images(state)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_mode not in ("disabled", "experimental", "native"):
            raise ValueError("image_mode must be 'disabled', 'experimental', or 'native'")

    def calibration_provenance(self, questions: Mapping[str, Question]) -> dict[str, Any]:
        """Identify SDK question translation and the effective judge service."""
        # SDK clients retain their resolved route in _config, rather than base_url.
        # Owned clients resolve explicit values before stripped environment/defaults.
        endpoint = (
            _sdk_client_api_root(self.client)
            if self.client is not None
            else self.base_url
            if self.base_url is not None
            else os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
        )
        image_input = any(
            isinstance(question.instructions, Mapping)
            and isinstance(
                modalities := question.instructions.get("required_modalities"), (list, tuple)
            )
            and "image" in modalities
            for question in questions.values()
        )
        return {
            "provider": "typesafe",
            "backend": "jev",
            "compiler_version": IMAGE_COMPILER_VERSION if image_input else COMPILER_VERSION,
            "endpoint": _normalize_api_root(endpoint),
            **(
                {
                    "image_mode": self.image_mode,
                    "input_serialization": "ordered-sdk-image-assets-v1",
                }
                if image_input
                else {}
            ),
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
            yield _SDKSession(self.client, self.model, self.image_mode)
            return
        async with AsyncTypeSafeClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            retry=RetryPolicy(max_retries=self.max_retries, timeout=self.timeout),
        ) as client:
            yield _SDKSession(client, self.model, self.image_mode)
