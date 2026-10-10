"""Native text/image Decisions adapter, with validation before SDK coercion.

The raw-response wrapper calls the official SDK's Decisions create method and
retains its transport/retry behavior. Inspecting the original JSON prevents
malformed wire values (for example, boolean probabilities) being coerced by SDK
models into apparently valid numbers.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from typed_evals.backends.jev import JudgeResponse, JudgeSession, Question
from typed_evals.data.models import ImageInput
from typed_evals.errors import InvalidAnswerError, OpenAIDecisionsError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.decision_create_params import Question as DecisionQuestion
    from openai.types.decision_input_message_param import DecisionInputMessageParam

DEFAULT_MODEL = "gpt-6-luna"
# Bump when serialization, question translation, or score interpretation changes.
COMPILER_VERSION = "1"
IMAGE_COMPILER_VERSION = "2"
MAX_IMAGES = 128


def serialize_state(state: dict[str, Any]) -> str:
    """Serialize exactly the supplied, already filtered evidence without truncation."""
    return json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _validate_images(state: Mapping[str, Any]) -> None:
    images = state.get("images")
    if images is None:
        return
    if not isinstance(images, (list, tuple)):
        raise ValueError("images must be a sequence of ImageInput evidence")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"Decisions accepts at most {MAX_IMAGES} images per request")


def compile_input(state: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Translate selected neutral evidence into native Decisions content parts.

    Text-only requests keep their original serialization. The text describes the
    order and labels of attached images, without duplicating image bytes as text.
    """
    _validate_images(state)
    images = state.get("images")
    if not images:
        return serialize_state(state)
    images = [ImageInput.model_validate(image) for image in images]
    text_state = {
        **state,
        "images": [
            {"index": index, "label": image.label, "detail": image.detail}
            for index, image in enumerate(images)
        ],
    }
    content = [{"type": "input_text", "text": serialize_state(text_state)}]
    content.extend(
        {"type": "input_image", "image_url": image.data_url, "detail": image.detail}
        for image in images
    )
    return [{"role": "user", "content": content}]


def compile_questions(questions: Mapping[str, Question]) -> list[dict[str, Any]]:
    """Translate TypeSafe schemas deterministically; retain ordered Score levels."""
    compiled: list[dict[str, Any]] = []
    names: set[str] = set()
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Decisions questions require unique nonempty string names")
        names.add(name)
        instructions = question.instructions
        if not isinstance(instructions, str):
            instructions = json.dumps(
                instructions, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        if question.type == "noul":
            predicate_criteria = question.criteria
            instructions += "\nEstimate the probability of the positive (true) condition, which represents a metric pass."
            if predicate_criteria is not None:
                # Metric validates descriptions as strings before creating SDK questions.
                instructions += "\nTrue condition: " + cast(str, predicate_criteria["true"])
                instructions += "\nFalse condition: " + cast(str, predicate_criteria["false"])
            compiled.append({"name": name, "type": "predicate", "instructions": instructions})
        elif question.type == "choice":
            choice_criteria = question.criteria
            if not isinstance(choice_criteria, dict) or not 2 <= len(choice_criteria) <= 255:
                raise ValueError("Decisions Choice requires between 2 and 255 options")
            if any(not isinstance(value, str) for value in choice_criteria):
                raise ValueError("Typed Evals Choice option values must be strings")
            compiled.append(
                {
                    "name": name,
                    "type": "choice",
                    "instructions": instructions,
                    "choices": [
                        {"value": value, "description": choice_criteria[value]}
                        for value in sorted(choice_criteria)
                    ],
                }
            )
        elif question.type == "score":
            score_criteria = question.criteria
            if not isinstance(score_criteria, (list, tuple)) or not 2 <= len(score_criteria) <= 10:
                raise ValueError("Typed Evals Score requires 2–10 ordered levels")
            compiled.append(
                {
                    "name": name,
                    "type": "score",
                    "instructions": instructions,
                    "levels": [
                        {"label": str(index), "description": description}
                        for index, description in enumerate(score_criteria)
                    ],
                }
            )
        else:
            raise ValueError("Unsupported Decisions question type")
    if not compiled:
        raise ValueError("Decisions requires at least one question")
    return compiled


def _probability(value: Any) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise InvalidAnswerError("Decisions probabilities must be finite numbers in [0, 1]")
    return float(value)


def _distribution(answer: dict[str, Any], question: dict[str, Any]) -> dict[str, float]:
    items = answer.get("probabilities")
    if not isinstance(items, list):
        raise InvalidAnswerError("Decisions probabilities must be an array")
    score = question["type"] == "score"
    expected = (
        {str(index) for index in range(len(question["levels"]))}
        if score
        else {option["value"] for option in question["choices"]}
    )
    probabilities = {}
    for item in items:
        if not isinstance(item, dict):
            raise InvalidAnswerError("Invalid Decisions probability entry")
        value = item.get("value")
        if score:
            if type(value) is not int or not 0 <= value < len(question["levels"]):
                raise InvalidAnswerError("Invalid Decisions ordinal index")
            key = str(value)
            if (
                type(item.get("label")) is not str
                or item["label"] != question["levels"][value]["label"]
            ):
                raise InvalidAnswerError("Decisions score label and ordinal index disagree")
        else:
            if type(value) is not str:
                raise InvalidAnswerError("Decisions Choice values must retain their string type")
            key = value
        if key in probabilities:
            raise InvalidAnswerError("Duplicate Decisions option or ordinal index")
        probabilities[key] = _probability(item.get("probability"))
    if set(probabilities) != expected:
        raise InvalidAnswerError("Decisions distribution does not match the requested rubric")
    if not math.isclose(
        sum(probabilities.values()), 1, abs_tol=min(0.025, 0.005 * len(probabilities) + 1e-6)
    ):
        raise InvalidAnswerError("Decisions probabilities do not sum approximately to one")
    return probabilities


def normalize_answer(answer: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """Validate one associated answer, then adapt to the existing metric parser."""
    if answer.get("type") != question["type"]:
        raise InvalidAnswerError("Decisions answer type differs from its question")
    if question["type"] == "predicate":
        return {"type": "noul", "noul": _probability(answer.get("probability"))}
    probabilities = _distribution(answer, question)
    confidence = _probability(answer.get("confidence"))
    if question["type"] == "choice":
        choice = answer.get("choice")
        if type(choice) is not str or choice not in probabilities:
            raise InvalidAnswerError("Invalid Decisions selected Choice value")
        if probabilities[choice] + 0.011 < max(probabilities.values()):
            raise InvalidAnswerError("Decisions selected Choice contradicts its distribution")
        return {
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    value = answer.get("score")
    levels = len(question["levels"]) - 1
    if type(value) not in (int, float):
        raise InvalidAnswerError("Invalid Decisions expected ordinal score")
    value = cast(int | float, value)
    if not 0 <= value <= levels:
        raise InvalidAnswerError("Invalid Decisions expected ordinal score")
    expected = sum(int(index) * probability for index, probability in probabilities.items())
    if abs(value - expected) > 0.01 + 0.005 * sum(range(levels + 1)):
        raise InvalidAnswerError("Decisions Score contradicts its distribution")
    # Metric.read_answer divides by levels exactly once. Do not normalize here.
    return {
        "type": "score",
        "score": value,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def normalize_answers(
    answers: Any, questions: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Literal["invalid_answer", "refusal"]]]:
    """Reject ambiguous associations globally; preserve attributable failures locally."""
    if not isinstance(answers, list):
        raise InvalidAnswerError("Decisions answers must be an array")
    expected = {question["name"]: question for question in questions}
    associated = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise InvalidAnswerError("Invalid Decisions answer structure")
        name = answer.get("name")
        if type(name) is not str or name not in expected or name in associated:
            raise InvalidAnswerError("Duplicate, missing, or unexpected Decisions question name")
        associated[name] = answer
    if set(associated) != set(expected):
        raise InvalidAnswerError("Decisions omitted requested question names")
    normalized: dict[str, dict[str, Any]] = {}
    errors: dict[str, Literal["invalid_answer", "refusal"]] = {}
    for name, question in expected.items():
        answer = associated[name]
        if answer.get("type") == "refusal":
            errors[name] = "refusal"
            continue
        try:
            normalized[name] = normalize_answer(answer, question)
        except InvalidAnswerError:
            errors[name] = "invalid_answer"
    return normalized, errors


def flatten_usage(usage: Any) -> dict[str, int | None]:
    """Flatten scalar token counters with dotted paths, retaining common top-level keys."""
    if not isinstance(usage, dict):
        raise InvalidAnswerError("Invalid Decisions usage structure")
    flattened: dict[str, int | None] = {}

    def visit(values: dict[str, Any], prefix: str = "") -> None:
        for key, value in values.items():
            if not isinstance(key, str) or "." in key:
                raise InvalidAnswerError("Invalid Decisions usage counter name")
            path = prefix + key
            if isinstance(value, dict):
                visit(value, path + ".")
            elif value is None or (type(value) is int and value >= 0):
                if path in flattened:
                    raise InvalidAnswerError("Duplicate Decisions usage counter")
                flattened[path] = value
            else:
                raise InvalidAnswerError("Invalid Decisions usage counter")

    visit(usage)
    return flattened


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Repeated JSON object keys must not erase an ambiguous wire value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate Decisions JSON field")
        result[key] = value
    return result


@dataclass
class _DecisionsSession:
    client: AsyncOpenAI
    model: str

    async def judge(
        self, state: dict[str, Any], questions: Mapping[str, Question]
    ) -> JudgeResponse:
        from openai import OpenAIError

        compiled = compile_questions(questions)
        try:
            raw = await self.client.decisions.with_raw_response.create(
                # The compiler constructs these wire shapes; the neutral dictionaries
                # remain useful for validation and provenance without SDK imports.
                model=self.model,
                input=cast("str | list[DecisionInputMessageParam]", compile_input(state)),
                questions=cast("list[DecisionQuestion]", compiled),
            )
        except OpenAIError as exc:
            # SDK errors can echo credentials/evidence. Export only the error
            # class, and suppress provider contents in the raised traceback.
            raise OpenAIDecisionsError(
                f"OpenAI Decisions request failed ({type(exc).__name__})"
            ) from None
        try:
            wire = raw.http_response.json(object_pairs_hook=_unique_json_object)
        except ValueError:
            raise InvalidAnswerError("Decisions returned invalid JSON") from None
        if (
            not isinstance(wire, dict)
            or type(wire.get("model")) is not str
            or not wire["model"].strip()
        ):
            raise InvalidAnswerError("Decisions returned an invalid model identity")
        answers, errors = normalize_answers(wire.get("answers"), compiled)
        return JudgeResponse(
            model=wire["model"],
            answers=answers,
            answer_errors=errors,
            usage=flatten_usage(wire.get("usage")),
        )


@dataclass
class OpenAIDecisionsBackend:
    """Use the optional OpenAI SDK. Injected async clients remain caller-owned."""

    supported_modalities: ClassVar[frozenset[str]] = frozenset({"text", "image"})

    model: str = DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 30.0
    max_retries: int = 2
    client: AsyncOpenAI | None = field(default=None, repr=False)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """Preflight provider limits for the whole batch before any request."""
        _validate_images(state)

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

    def calibration_provenance(self, questions: Mapping[str, Question]) -> dict[str, Any]:
        image_input = any(
            isinstance(question.instructions, Mapping)
            and "image" in cast(Sequence[str], question.instructions.get("required_modalities", ()))
            for question in questions.values()
        )
        # A custom route can return the same model string. Bind curves to the
        # effective endpoint without URL userinfo, query, or fragment credentials.
        endpoint = (
            str(self.client.base_url)
            if self.client is not None
            else os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        try:
            parsed = urlsplit(endpoint)
            host = parsed.hostname
            port = parsed.port
            if parsed.scheme not in ("https", "http") or not host:
                raise ValueError
            if ":" in host:
                host = f"[{host}]"
            if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
                host += f":{port}"
            endpoint = urlunsplit((parsed.scheme, host, parsed.path.rstrip("/") + "/", "", ""))
        except ValueError:
            raise ValueError("Decisions requires a valid HTTP(S) endpoint for provenance") from None
        return {
            "provider": "openai",
            "backend": "openai-decisions",
            "compiler_version": IMAGE_COMPILER_VERSION if image_input else COMPILER_VERSION,
            "endpoint": endpoint,
            "input_serialization": "json-unicode-sorted-compact+ordered-image-parts-v1"
            if image_input
            else "json-unicode-sorted-compact",
            "questions": compile_questions(questions),
        }

    @asynccontextmanager
    async def session(self) -> AsyncIterator[JudgeSession]:
        # Imports, including runtime annotations, never require the optional extra.
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "Install Decisions support with pip install 'typed_evals[openai]'"
            ) from None
        if self.client is not None:
            if not hasattr(self.client, "decisions"):
                raise ImportError(
                    "Decisions requires OpenAI SDK >=3.26.0; pip install 'typed_evals[openai]'"
                )
            yield _DecisionsSession(self.client, self.model)
            return
        async with AsyncOpenAI(
            api_key=self.api_key, timeout=self.timeout, max_retries=self.max_retries
        ) as client:
            if not hasattr(client, "decisions"):
                raise ImportError(
                    "Decisions requires OpenAI SDK >=3.26.0; pip install 'typed_evals[openai]'"
                )
            yield _DecisionsSession(client, self.model)
