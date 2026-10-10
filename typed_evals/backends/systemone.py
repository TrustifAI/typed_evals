"""Provider-neutral HTTP adapter for Jev-compatible System One servers.

Model weights and inference runtimes belong to the server. This adapter only
exchanges typed questions and answers; it never generates or parses judge prose.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

import httpx2

from typed_evals.backends.capabilities import validate_evidence
from typed_evals.backends.jev import JudgeResponse, JudgeSession, Question
from typed_evals.errors import InvalidAnswerError, SystemOneError

COMPILER_VERSION = "1"


def _endpoint(base_url: str) -> str:
    """Require an explicit API root and normalize its transport identity."""
    try:
        if not isinstance(base_url, str) or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in base_url
        ):
            raise ValueError
        parsed = urlsplit(base_url)
        host, port = parsed.hostname, parsed.port
        if (
            parsed.scheme not in ("http", "https")
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        if ":" in host:
            host = f"[{host}]"
        if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
            host += f":{port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/") + "/v1/systemone", "", ""))
    except ValueError:
        raise ValueError(
            "base_url must be an HTTP(S) API root without credentials, query, or fragment"
        ) from None


def _compile_questions(questions: Mapping[str, Question]) -> dict[str, dict[str, Any]]:
    if not questions:
        raise ValueError("System One requires at least one question")
    compiled = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("System One question names must be nonempty strings")
        # Keep structured instructions, evaluation policy, and rubric order intact.
        compiled[name] = question.model_dump(mode="json")
    return compiled


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate System One JSON field")
        result[key] = value
    return result


def _usage(value: Any) -> dict[str, int | None]:
    if not isinstance(value, dict):
        raise InvalidAnswerError("System One returned invalid usage")
    counters = {}
    for key, counter in value.items():
        if type(counter) is int and counter >= 0 or counter is None:
            counters[key] = counter
        elif key in {"input_tokens", "output_tokens"}:
            raise InvalidAnswerError("System One returned an invalid token counter")
        # Some servers (e.g. Laya) include diagnostic booleans/lists. They are
        # metadata, not scalar usage counters, and do not belong in JudgeResponse.
    return counters


@dataclass
class _SystemOneSession:
    client: httpx2.AsyncClient = field(repr=False)
    model: str
    endpoint: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    max_retries: int
    supported_modalities: ClassVar[frozenset[str]] = frozenset({"text"})

    async def judge(
        self, state: dict[str, Any], questions: Mapping[str, Question]
    ) -> JudgeResponse:
        validate_evidence(self, state)
        body = {"model": self.model, "state": state, "questions": _compile_questions(questions)}
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(self.endpoint, json=body, headers=self.headers)
            except httpx2.TransportError as exc:
                if attempt == self.max_retries:
                    raise SystemOneError(
                        f"System One request failed ({type(exc).__name__})"
                    ) from None
            else:
                if response.is_success:
                    break
                retryable = response.status_code in {408, 409, 429} or (
                    500 <= response.status_code < 600
                )
                if not retryable or attempt == self.max_retries:
                    raise SystemOneError(
                        f"System One request failed (HTTP {response.status_code})"
                    ) from None
            await asyncio.sleep(min(0.5 * 2 ** min(attempt, 4), 8.0))

        try:
            wire = response.json(object_pairs_hook=_unique_object)
        except ValueError:
            raise InvalidAnswerError("System One returned invalid JSON") from None
        if (
            not isinstance(wire, dict)
            or not isinstance(wire.get("model"), str)
            or not wire["model"].strip()
        ):
            raise InvalidAnswerError("System One returned an invalid model identity")
        supplied = wire.get("answers")
        if not isinstance(supplied, dict) or set(supplied) - set(questions):
            raise InvalidAnswerError("System One returned invalid answer names")
        answers = {}
        errors: dict[str, Any] = {}
        for name in questions:
            answer = supplied.get(name)
            if isinstance(answer, dict):
                # Preserve wire values: Metric.read_answer rejects coercible strings,
                # booleans, invalid distributions, and contradictory selected values.
                answers[name] = answer
            else:
                errors[name] = "invalid_answer"
        return JudgeResponse(
            model=wire["model"],
            answers=answers,
            answer_errors=errors,
            usage=_usage(wire.get("usage", {})),
        )


@dataclass
class SystemOneBackend:
    """Connect to any text-only Jev-compatible ``/v1/systemone`` server.

    ``base_url`` is the API root, including any gateway path prefix. ``model``
    identifies a model deployed there; a Hugging Face repo ID alone does not
    deploy weights. Authentication is optional and explicit: no TypeSafe
    environment variables or credentials are inherited.

    One HTTP client pools a batch's requests. Injected ``httpx2.AsyncClient``
    instances retain their timeout configuration and remain caller-owned; this
    adapter owns its bounded request retries, endpoint, and optional bearer key.
    """

    model: str
    base_url: str = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 30.0
    max_retries: int = 2
    client: httpx2.AsyncClient | None = field(default=None, repr=False)
    supported_modalities: ClassVar[frozenset[str]] = frozenset({"text"})

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty identifier")
        _endpoint(self.base_url)
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
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key
            or not self.api_key.isascii()
            or any(char.isspace() or ord(char) < 33 or ord(char) == 127 for char in self.api_key)
        ):
            raise ValueError("api_key must be a nonempty ASCII bearer token without whitespace")

    def calibration_provenance(self, questions: Mapping[str, Question]) -> dict[str, Any]:
        compiled = _compile_questions(questions)
        return {
            "provider": "systemone",
            "backend": "systemone",
            "compiler_version": COMPILER_VERSION,
            "endpoint": _endpoint(self.base_url),
            "questions": [
                {
                    "name": name,
                    "schema": schema,
                    "criteria_order": list(schema["criteria"])
                    if isinstance(schema.get("criteria"), dict)
                    else None,
                }
                for name, schema in compiled.items()
            ],
        }

    @asynccontextmanager
    async def session(self) -> AsyncIterator[JudgeSession]:
        endpoint = _endpoint(self.base_url)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key is not None else {}
        if self.client is not None:
            yield _SystemOneSession(self.client, self.model, endpoint, headers, self.max_retries)
            return
        async with httpx2.AsyncClient(timeout=self.timeout) as client:
            yield _SystemOneSession(client, self.model, endpoint, headers, self.max_retries)
