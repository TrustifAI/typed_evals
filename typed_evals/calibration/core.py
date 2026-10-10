"""Binary event calibration with portable Venn-Abers and isotonic predictors."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from typed_evals._utils import digest, write_json
from typed_evals.backends.provenance import validate_provenance
from typed_evals.data.models import EvaluationSample, Model, Probability
from typed_evals.errors import CalibrationError, CalibrationMismatchError, DataLeakageError
from typed_evals.metrics import EVIDENCE_FIELDS, Metric


class CalibrationConfig(Model):
    enabled: bool = False
    algorithm: Literal["auto", "venn_abers", "isotonic"] = "auto"
    isotonic_min_samples: Annotated[int, Field(ge=2, strict=True)] = 2000
    validation_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.2
    min_samples: Annotated[int, Field(ge=4, strict=True)] = 100
    min_validation_samples: Annotated[int, Field(ge=2, strict=True)] = 20
    min_class_samples: Annotated[int, Field(ge=1, strict=True)] = 2
    random_state: int = 42
    n_bins: Annotated[int, Field(ge=2, le=100, strict=True)] = 10

    def algorithm_for(self, training_samples: int) -> Literal["venn_abers", "isotonic"]:
        if self.algorithm != "auto":
            return self.algorithm
        return "isotonic" if training_samples >= self.isotonic_min_samples else "venn_abers"


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
        predicted: float | None
        observed: float | None
        if n:
            predicted = sum(predictions[position] for position in selected) / n
            observed = sum(labels[position] for position in selected) / n
            ece += n / len(labels) * abs(predicted - observed)
        else:
            predicted = observed = None
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


def _validate_score(score: float) -> None:
    if (
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise CalibrationError("raw score must be finite and in [0, 1]")


class IsotonicCalibrator(Model):
    algorithm: Literal["isotonic"] = "isotonic"
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
        _validate_score(score)
        # Same linear interpolation and endpoint clipping as sklearn predict().
        if score <= self.x[0]:
            return self.y[0]
        if score >= self.x[-1]:
            return self.y[-1]
        right = bisect.bisect_right(self.x, score)
        left = right - 1
        fraction = (score - self.x[left]) / (self.x[right] - self.x[left])
        return self.y[left] + fraction * (self.y[right] - self.y[left])


def _venn_abers_upper(counts: Sequence[int], positives: Sequence[int]) -> tuple[float, ...]:
    """Precompute label-1 predictions via the moving greatest convex minorant.

    Algorithms 1-2 of Vovk, Petej and Fedorova (2015), arXiv:1511.00213.
    Each point is pushed/popped at most once, so this scan is linear after sorting.
    Integer cumulative sums avoid roundoff in the hull orientation tests.
    """
    points = [(-1, -1), (0, 0)]
    for count, positive in zip(counts, positives, strict=True):
        total, successes = points[-1]
        points.append((total + count, successes + positive))

    def cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
        return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

    hull: list[tuple[int, int]] = []
    for point in points:
        while len(hull) > 1 and cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    stack = hull[::-1]
    predictions = []
    for index in range(1, len(counts) + 1):
        left, right = stack[-1], stack[-2]
        predictions.append((right[1] - left[1]) / (right[0] - left[0]))
        # Swap the hypothetical positive observation past the next score group.
        previous, current, following = points[index - 1 : index + 2]
        moved = (
            previous[0] + following[0] - current[0],
            previous[1] + following[1] - current[1],
        )
        points[index] = moved
        if cross(left, right, moved) >= 0:
            continue
        stack.pop()
        while len(stack) > 1 and cross(moved, stack[-1], stack[-2]) <= 0:
            stack.pop()
        stack.append(moved)
    return tuple(predictions)


class VennAbersCalibrator(Model):
    """Inductive Venn-Abers pair with a log-loss minimax point prediction.

    The pair comes from isotonic fits augmented with the query labeled 0 and 1.
    Stored step tables handle tied scores exactly, without refitting at inference.
    The pair is not a confidence interval; its validity does not transfer to the
    single probability returned by predict().
    """

    algorithm: Literal["venn_abers"] = "venn_abers"
    x: tuple[Probability, ...]
    p0: tuple[Probability, ...]
    p1: tuple[Probability, ...]
    n_samples: Annotated[int, Field(ge=2, strict=True)]

    @model_validator(mode="after")
    def validate_tables(self) -> VennAbersCalibrator:
        if not self.x or len(self.x) != len(self.p0) or len(self.x) != len(self.p1):
            raise ValueError("Venn-Abers tables must be nonempty with equal lengths")
        if len(self.x) > self.n_samples:
            raise ValueError("Venn-Abers cannot have more score groups than training samples")
        if any(a >= b for a, b in zip(self.x, self.x[1:], strict=False)):
            raise ValueError("Venn-Abers x knots must strictly increase")
        for values in (self.p0, self.p1):
            if any(a > b for a, b in zip(values, values[1:], strict=False)):
                raise ValueError("Venn-Abers probabilities must never decrease")
        if any(a >= b for a, b in zip(self.p0, self.p1, strict=True)):
            raise ValueError("Venn-Abers requires p0 < p1 at each score")
        if any(a >= b for a, b in zip(self.p0[:-1], self.p1[1:], strict=True)):
            raise ValueError("Venn-Abers requires p0 < p1 between scores")
        return self

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[int]) -> VennAbersCalibrator:
        _validate_xy(scores, labels)
        if len(set(labels)) != 2:
            raise CalibrationError(
                "Venn-Abers fitting requires examples of both passing and failing"
            )
        groups: dict[float, tuple[int, int]] = {}
        for score, label in zip(scores, labels, strict=True):
            count, positive = groups.get(float(score), (0, 0))
            groups[float(score)] = (count + 1, positive + int(label))
        x = tuple(sorted(groups))
        counts = [groups[score][0] for score in x]
        positives = [groups[score][1] for score in x]
        p1 = _venn_abers_upper(counts, positives)
        # Mirror scores and complement labels to reuse the label-1 scan for p0.
        reflected = _venn_abers_upper(
            counts[::-1],
            [count - positive for count, positive in zip(counts, positives, strict=True)][::-1],
        )
        return cls(
            x=x,
            p0=tuple(1 - value for value in reversed(reflected)),
            p1=p1,
            n_samples=len(labels),
        )

    def predict_interval(self, score: float) -> tuple[float, float]:
        _validate_score(score)
        lower = bisect.bisect_right(self.x, score) - 1
        upper = bisect.bisect_left(self.x, score)
        return (
            self.p0[lower] if lower >= 0 else 0.0,
            self.p1[upper] if upper < len(self.x) else 1.0,
        )

    def predict(self, score: float) -> float:
        p0, p1 = self.predict_interval(score)
        return p1 / (1 - p0 + p1)


CalibrationCurve = Annotated[
    IsotonicCalibrator | VennAbersCalibrator, Field(discriminator="algorithm")
]


class MetricCalibrationReport(Model):
    algorithm: Literal["isotonic", "venn_abers"] = "isotonic"
    training_samples: int
    validation_samples: int
    training_positive_rate: Probability
    raw: ProbabilityDiagnostics
    calibrated: ProbabilityDiagnostics
    brier_improved: bool
    log_loss_improved: bool = False
    notes: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def legacy_log_loss_flag(cls, value: Any) -> Any:
        if isinstance(value, dict) and "log_loss_improved" not in value:
            losses = []
            for name in ("raw", "calibrated"):
                diagnostics = value.get(name)
                loss = (
                    diagnostics.log_loss
                    if isinstance(diagnostics, ProbabilityDiagnostics)
                    else diagnostics.get("log_loss")
                    if isinstance(diagnostics, dict)
                    else None
                )
                losses.append(loss)
            raw_loss, calibrated_loss = losses
            if isinstance(raw_loss, Real) and isinstance(calibrated_loss, Real):
                value = {**value, "log_loss_improved": calibrated_loss < raw_loss}
        return value


class CalibrationReport(Model):
    split: Literal["automatic_group_holdout", "explicit_holdout"]
    random_state: int
    metrics: dict[str, MetricCalibrationReport]
    note: str = "Diagnostics use held-out rows. Deployed curves are fitted only on training rows."


class CalibrationBundle(Model):
    # Keep the default for manually constructed legacy bundles; new fits set v3 explicitly.
    schema_version: Literal[1, 2, 3] = 1
    target: Literal["metric_pass"] = "metric_pass"
    created_at: str
    requested_model: str
    observed_model: str
    metric_fingerprints: dict[str, str]
    metric_order: tuple[str, ...]
    evidence_fields: tuple[str, ...]
    curves: dict[str, CalibrationCurve]
    report: CalibrationReport
    reserved_sample_hashes: frozenset[str]
    reserved_group_hashes: frozenset[str] = frozenset()
    backend_provenance: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def serialize_bundle(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if self.schema_version < 3:
            # Re-saving a legacy artifact keeps its original JSON schema, so old
            # readers do not receive unknown v3-only fields under a v1/v2 tag.
            for curve in value.get("curves", {}).values():
                curve.pop("algorithm", None)
            for report in value.get("report", {}).get("metrics", {}).values():
                report.pop("algorithm", None)
                report.pop("log_loss_improved", None)
        return value

    @model_validator(mode="before")
    @classmethod
    def legacy_curve_tags(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("schema_version", 1) in (1, 2):
            curves = value.get("curves")
            if isinstance(curves, dict):
                value = {
                    **value,
                    "curves": {
                        name: {"algorithm": "isotonic", **curve}
                        if isinstance(curve, dict)
                        else curve
                        for name, curve in curves.items()
                    },
                }
        return value

    @model_validator(mode="after")
    def complete_bundle(self) -> CalibrationBundle:
        if self.schema_version == 1:
            if self.backend_provenance is not None:
                raise ValueError("Legacy v1 artifacts cannot claim backend provenance")
        else:
            if self.backend_provenance is None:
                raise ValueError("Version 2/3 artifacts require explicit backend provenance")
            validate_provenance(self.backend_provenance)
        names = set(self.metric_fingerprints)
        if not names or names != set(self.curves) or names != set(self.report.metrics):
            raise ValueError("artifact metrics, curves, fingerprints, and report must agree")
        for name, curve in self.curves.items():
            if self.schema_version < 3 and curve.algorithm != "isotonic":
                raise ValueError("Legacy artifacts support only isotonic curves")
            if self.report.metrics[name].algorithm != curve.algorithm:
                raise ValueError("artifact curve and reported algorithm must agree")
        if set(self.metric_order) != names or len(self.metric_order) != len(names):
            raise ValueError("artifact metric_order must contain each metric exactly once")
        if not self.evidence_fields or set(self.evidence_fields) - EVIDENCE_FIELDS:
            raise ValueError("artifact has invalid evidence fields")
        if not self.requested_model or not self.observed_model or not self.reserved_sample_hashes:
            raise ValueError("artifact is missing its model identity or dataset hashes")
        return self

    def validate_for(
        self,
        metrics: Sequence[Metric],
        requested_model: str,
        *,
        backend_provenance: dict[str, Any] | None = None,
    ) -> None:
        # Retain the two-argument legacy/custom check. A verified v2/v3 artifact needs a
        # current backend identity; evaluation and loading always provide one.
        if backend_provenance is not None:
            self._validate_backend(backend_provenance)
        elif (
            self.schema_version >= 2
            and cast(dict[str, Any], self.backend_provenance)["status"] == "verified"
        ):
            raise CalibrationMismatchError(
                "Verified calibration requires current backend provenance; use an evaluator"
            )
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

    def _validate_backend(self, provenance: dict[str, Any]) -> None:
        validate_provenance(provenance)
        if self.schema_version == 1:
            configuration = provenance.get("configuration", {})
            if provenance["status"] == "unknown" or (
                configuration.get("provider") == "typesafe"
                and configuration.get("backend") == "jev"
                and configuration.get("compiler_version") == "1"
            ):
                # Established v1 Jev and model/session-only custom workflows remain usable.
                # This compatibility rule asserts no historical provider/compiler identity.
                return
            raise CalibrationMismatchError(
                "Legacy calibration has no backend/compiler provenance; refit with this backend"
            )
        if provenance != self.backend_provenance:
            raise CalibrationMismatchError(
                "Calibration backend, compiler, or score-producing configuration changed; refit"
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
