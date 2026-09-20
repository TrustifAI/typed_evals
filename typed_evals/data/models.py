from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictStr, field_validator

from typed_evals._utils import digest, write_json

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ToolCall(Model):
    """An observed execution event. Never put hidden chain-of-thought in a trace."""

    name: Annotated[StrictStr, Field(min_length=1)]
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    output: JsonValue = None
    status: Literal["success", "error", "unknown"] = "unknown"
    error: StrictStr | None = None


class ToolProposal(Model):
    """A requested call, before execution. It cannot contain a claimed result."""

    name: Annotated[StrictStr, Field(min_length=1)]
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool name must contain text")
        return value


class EvaluationSample(Model):
    input: Annotated[StrictStr, Field(min_length=1)]
    response: StrictStr
    contexts: tuple[StrictStr, ...] = ()
    reference: StrictStr | None = None
    trace: tuple[ToolCall, ...] = ()
    expected_outcome: StrictStr | None = None
    proposed_tool_call: ToolProposal | None = None
    id: StrictStr | None = None
    group_id: StrictStr | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def nonblank_input(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must contain text")
        return value

    @field_validator("contexts")
    @classmethod
    def nonblank_contexts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("contexts must not contain blank passages")
        return value

    def state(self, fields: tuple[str, ...] | set[str]) -> dict[str, Any]:
        return self.model_dump(mode="json", include=set(fields))

    @property
    def content_hash(self) -> str:
        # IDs, grouping and metadata are never evidence for a metric.
        excluded = {"id", "group_id", "metadata"}
        if self.proposed_tool_call is None:
            # Preserve identities of existing offline samples and calibration data.
            excluded.add("proposed_tool_call")
        return digest(self.model_dump(mode="json", exclude=excluded))

    @property
    def group_hash(self) -> str | None:
        return digest(self.group_id) if self.group_id is not None else None


class CalibrationExample(Model):
    sample: EvaluationSample
    labels: dict[str, int]

    @field_validator("labels", mode="before")
    @classmethod
    def binary_labels(cls, value: Any) -> dict[str, int]:
        if not isinstance(value, dict) or not value:
            raise ValueError("labels must be a nonempty mapping of metric names to 0 or 1")
        for key, label in value.items():
            if not isinstance(key, str) or type(label) not in (int, bool) or label not in (0, 1):
                raise ValueError("human labels must be integer 0/1 or boolean, not soft scores")
        return {key: int(label) for key, label in value.items()}


class MetricResult(Model):
    name: str
    status: Literal["ok", "skipped", "error"]
    raw_score: Probability | None = None
    calibrated_probability: Probability | None = None
    raw_kind: Literal["event_probability", "normalized_ordinal_score"] | None = None
    threshold: Probability
    passed: bool | None = None
    confidence: Probability | None = None
    probabilities: dict[str, float] | None = None
    selected_choice: str | None = None
    calibration_target: Literal["metric_pass"] | None = None
    message: str | None = None

    @property
    def score(self) -> float | None:
        if self.calibrated_probability is not None:
            return self.calibrated_probability
        return self.raw_score


class SampleResult(Model):
    sample_id: str
    tool_name: str | None = Field(
        default=None, description="Name of the proposed tool being evaluated, when supplied."
    )
    sample_hash: str
    model: str | None
    metrics: dict[str, MetricResult]
    elapsed_ms: float
    usage: dict[str, int | None] = Field(default_factory=dict)

    @property
    def passed(self) -> bool | None:
        if not self.metrics or any(item.status != "ok" for item in self.metrics.values()):
            return None
        return all(item.passed for item in self.metrics.values())


class MetricSummary(Model):
    evaluated: int
    skipped: int
    errors: int
    passed: int
    mean_raw_score: float | None
    mean_score: float | None
    pass_rate: Probability | None


class EvaluationReport(Model):
    results: tuple[SampleResult, ...]
    summary: dict[str, MetricSummary]
    calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        for row, result in zip(data["results"], self.results, strict=True):
            row["passed"] = result.passed
            for name, metric in result.metrics.items():
                row["metrics"][name]["score"] = metric.score
        return data

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())
