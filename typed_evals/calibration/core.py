"""Binary event calibration; sklearn is imported only when fitting a curve."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from typed_evals._utils import digest, write_json
from typed_evals.data.models import EvaluationSample, Model, Probability
from typed_evals.errors import CalibrationError, CalibrationMismatchError, DataLeakageError
from typed_evals.metrics import EVIDENCE_FIELDS, Metric


class CalibrationConfig(Model):
    enabled: bool = False
    validation_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.2
    min_samples: Annotated[int, Field(ge=4, strict=True)] = 100
    min_validation_samples: Annotated[int, Field(ge=2, strict=True)] = 20
    min_class_samples: Annotated[int, Field(ge=1, strict=True)] = 2
    random_state: int = 42
    n_bins: Annotated[int, Field(ge=2, le=100, strict=True)] = 10


class ReliabilityBin(Model):
    lower: float
    upper: float
    count: int
    mean_prediction: Probability | None
    observed_positive_rate: Probability | None


class ProbabilityDiagnostics(Model):
    count: int
    brier: float
    log_loss: float
    ece: float
    accuracy_at_half: Probability
    bins: tuple[ReliabilityBin, ...]


def probability_diagnostics(
    predictions: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> ProbabilityDiagnostics:
    if type(n_bins) is not int or not 2 <= n_bins <= 100:
        raise CalibrationError("n_bins must be between 2 and 100")
    _validate_xy(predictions, labels)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        selected = [
            position
            for position, value in enumerate(predictions)
            if min(int(value * n_bins), n_bins - 1) == index
        ]
        n = len(selected)
        predicted = sum(predictions[position] for position in selected) / n if n else None
        observed = sum(labels[position] for position in selected) / n if n else None
        if n:
            ece += n / len(labels) * abs(predicted - observed)
        bins.append(
            ReliabilityBin(
                lower=index / n_bins,
                upper=(index + 1) / n_bins,
                count=n,
                mean_prediction=predicted,
                observed_positive_rate=observed,
            )
        )
    clipped = [min(1 - 1e-15, max(1e-15, value)) for value in predictions]
    return ProbabilityDiagnostics(
        count=len(labels),
        brier=sum((value - label) ** 2 for value, label in zip(predictions, labels, strict=True))
        / len(labels),
        log_loss=-sum(
            label * math.log(value) + (1 - label) * math.log1p(-value)
            for value, label in zip(clipped, labels, strict=True)
        )
        / len(labels),
        ece=ece,
        accuracy_at_half=sum(
            int(value >= 0.5) == label for value, label in zip(predictions, labels, strict=True)
        )
        / len(labels),
        bins=tuple(bins),
    )


def _validate_xy(scores: Sequence[float], labels: Sequence[int]) -> None:
    if len(scores) != len(labels) or not len(scores):
        raise CalibrationError("scores and labels must have the same nonzero length")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in scores
    ):
        raise CalibrationError("raw scores must be finite values in [0, 1]")
    if any(not isinstance(label, Integral) or label not in (0, 1) for label in labels):
        raise CalibrationError("labels must be binary 0/1")


class IsotonicCalibrator(Model):
    x: tuple[Probability, ...]
    y: tuple[Probability, ...]
    n_samples: Annotated[int, Field(ge=2, strict=True)]

    @model_validator(mode="after")
    def validate_knots(self) -> IsotonicCalibrator:
        if not self.x or len(self.x) != len(self.y):
            raise ValueError("isotonic knot arrays must be nonempty with equal lengths")
        if any(left >= right for left, right in zip(self.x, self.x[1:], strict=False)):
            raise ValueError("isotonic x knots must strictly increase")
        if any(left > right for left, right in zip(self.y, self.y[1:], strict=False)):
            raise ValueError("isotonic y knots must never decrease")
        return self

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        _validate_xy(scores, labels)
        if len(set(labels)) != 2:
            raise CalibrationError("isotonic fitting requires examples of both passing and failing")
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise CalibrationError(
                "Install calibration dependencies: pip install 'typed_evals[calibration]'"
            ) from exc
        model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
        model.fit(scores, labels)
        return cls(
            x=tuple(float(value) for value in model.X_thresholds_),
            y=tuple(float(value) for value in model.y_thresholds_),
            n_samples=len(labels),
        )

    def predict(self, score: float) -> float:
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise CalibrationError("raw score must be finite and in [0, 1]")
        # Same linear interpolation and endpoint clipping as sklearn predict().
        if score <= self.x[0]:
            return self.y[0]
        if score >= self.x[-1]:
            return self.y[-1]
        right = bisect.bisect_right(self.x, score)
        left = right - 1
        fraction = (score - self.x[left]) / (self.x[right] - self.x[left])
        return self.y[left] + fraction * (self.y[right] - self.y[left])


class MetricCalibrationReport(Model):
    training_samples: int
    validation_samples: int
    training_positive_rate: Probability
    raw: ProbabilityDiagnostics
    calibrated: ProbabilityDiagnostics
    brier_improved: bool
    notes: tuple[str, ...] = ()


class CalibrationReport(Model):
    split: Literal["automatic_group_holdout", "explicit_holdout"]
    random_state: int
    metrics: dict[str, MetricCalibrationReport]
    note: str = "Diagnostics use held-out rows. Deployed curves are fitted only on training rows."


class CalibrationBundle(Model):
    schema_version: Literal[1] = 1
    target: Literal["metric_pass"] = "metric_pass"
    created_at: str
    requested_model: str
    observed_model: str
    metric_fingerprints: dict[str, str]
    metric_order: tuple[str, ...]
    evidence_fields: tuple[str, ...]
    curves: dict[str, IsotonicCalibrator]
    report: CalibrationReport
    reserved_sample_hashes: frozenset[str]
    reserved_group_hashes: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def complete_bundle(self) -> CalibrationBundle:
        names = set(self.metric_fingerprints)
        if not names or names != set(self.curves) or names != set(self.report.metrics):
            raise ValueError("artifact metrics, curves, fingerprints, and report must agree")
        if set(self.metric_order) != names or len(self.metric_order) != len(names):
            raise ValueError("artifact metric_order must contain each metric exactly once")
        if not self.evidence_fields or set(self.evidence_fields) - EVIDENCE_FIELDS:
            raise ValueError("artifact has invalid evidence fields")
        if not self.requested_model or not self.observed_model or not self.reserved_sample_hashes:
            raise ValueError("artifact is missing its model identity or dataset hashes")
        return self

    def validate_for(self, metrics: Sequence[Metric], requested_model: str) -> None:
        if requested_model != self.requested_model:
            raise CalibrationMismatchError(
                "Requested model differs from the calibration model; refit"
            )
        if tuple(metric.name for metric in metrics) != self.metric_order:
            raise CalibrationMismatchError("Metric panel or order changed; refit calibration")
        if {metric.name: metric.fingerprint for metric in metrics} != self.metric_fingerprints:
            raise CalibrationMismatchError(
                "Metric wording, criteria, fields, or version changed; refit"
            )
        fields = tuple(sorted({field for metric in metrics for field in metric.required_fields}))
        if fields != self.evidence_fields:
            raise CalibrationMismatchError(
                "Calibration evidence fields do not match the metric panel"
            )

    def check_model(self, model: str) -> None:
        if model != self.observed_model:
            raise CalibrationMismatchError(
                "The API returned a different model version; refit calibration"
            )

    def check_overlap(self, samples: Sequence[EvaluationSample]) -> None:
        for sample in samples:
            if digest(sample.state(self.evidence_fields)) in self.reserved_sample_hashes:
                raise DataLeakageError(
                    "Evaluation reuses calibration/validation content; use a separate test set"
                )
            if sample.group_hash is not None and sample.group_hash in self.reserved_group_hashes:
                raise DataLeakageError("Evaluation group overlaps calibration/validation groups")

    def save(self, path: str | Path) -> None:
        write_json(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationBundle:
        try:
            return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            raise CalibrationError("Invalid or incompatible calibration JSON artifact") from exc
