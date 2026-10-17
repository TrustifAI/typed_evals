"""Compare an evaluator with caller-supplied labels without fitting or tuning it."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any, Literal

from typed_evals._utils import digest, run_sync, write_json
from typed_evals.calibration import (
    CalibrationBundle,
    ProbabilityDiagnostics,
    probability_diagnostics,
)
from typed_evals.data.models import CalibrationExample, Model, Probability
from typed_evals.errors import DataLeakageError, MissingInputError
from typed_evals.evaluation.evaluator import Evaluator


class MetricValidationReport(Model):
    """Diagnostics for labeled rows only; unavailable judgments never become scores.

    Accuracy, AUROC, and the majority-class baseline use the same evaluated subset.
    The baseline is descriptive: it is the larger class fraction in that subset,
    not a predictor learned on independent training data. Coverage uses all labels.
    """

    labeled: int
    evaluated: int
    unavailable: int
    skipped: int
    errors: int
    coverage: Probability | None
    threshold: Probability
    raw_kind: Literal["event_probability", "normalized_ordinal_score", "relevance_fraction"]
    effective_kind: Literal["event_probability", "normalized_ordinal_score", "relevance_fraction"]
    raw: ProbabilityDiagnostics | None
    effective: ProbabilityDiagnostics | None
    raw_roc_auc: Probability | None
    effective_roc_auc: Probability | None
    majority_baseline_accuracy: Probability | None


class ValidationReport(Model):
    """An evaluation against supplied labels, with no claim about label authenticity."""

    examples: int
    calibrated: bool
    label_source: Literal["caller_supplied"] = "caller_supplied"
    metrics: dict[str, MetricValidationReport]
    note: str = (
        "Labels are caller supplied; their quality, representativeness, and independence "
        "are not verified. Diagnostics use available labeled judgments. No fitting or "
        "threshold selection is performed. For normalized_ordinal_score and relevance_fraction, "
        "Brier, ECE, and "
        "log loss treat the score as a numeric probability proxy; they do not establish "
        "event-probability semantics."
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())


def validate_metrics(
    examples: Sequence[CalibrationExample],
    *,
    evaluator: Evaluator,
    calibration: CalibrationBundle | None = None,
) -> ValidationReport:
    """Evaluate labeled examples at the evaluator's existing metric thresholds.

    Supply a separate, representative test set after choosing rubrics, calibration,
    and thresholds. A calibration bundle's reserved content/group overlap checks
    remain enforced. Labels are never sent to the judge. Sparse labels are allowed;
    unlabeled metric judgments do not contribute to that metric's diagnostics.
    Repeated selected-panel evidence is rejected, regardless of IDs or labels.
    """
    return run_sync(
        lambda: avalidate_metrics(examples, evaluator=evaluator, calibration=calibration)
    )


async def avalidate_metrics(
    examples: Sequence[CalibrationExample],
    *,
    evaluator: Evaluator,
    calibration: CalibrationBundle | None = None,
) -> ValidationReport:
    """Async counterpart to :func:`validate_metrics`, honoring evaluator error policies."""
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples must contain at least one CalibrationExample")
    if any(not isinstance(row, CalibrationExample) for row in examples):
        raise TypeError("examples must contain CalibrationExample objects")
    metric_map = {metric.name: metric for metric in evaluator.metrics}
    for index, row in enumerate(examples):
        unknown = set(row.labels) - metric_map.keys()
        if unknown:
            raise ValueError(f"Unknown metric labels: {sorted(unknown)}")
        for name in row.labels:
            absent = metric_map[name].missing_fields(row.sample)
            if absent and evaluator.missing == "raise":
                raise MissingInputError(
                    f"Validation row {index}: {name} lacks required sample evidence "
                    f"({', '.join(absent)})"
                )
    evidence_fields = {field for metric in evaluator.metrics for field in metric.required_fields}
    hashes = [digest(row.sample.state(evidence_fields)) for row in examples]
    if len(set(hashes)) != len(hashes):
        raise DataLeakageError(
            "Validation data contains repeated selected-panel evidence; merge sparse labels "
            "and resolve label conflicts before validating"
        )
    # Preserve the complete shared panel, even on sparsely labeled rows. Filtering
    # the panel per label would change the judge request and invalidate calibration.
    evaluation = await evaluator.aevaluate(
        [row.sample for row in examples], calibration=calibration
    )
    metrics = {}
    for name, metric in metric_map.items():
        labels, raw, effective = [], [], []
        labeled = skipped = errors = 0
        for row, result in zip(examples, evaluation.results, strict=True):
            if name not in row.labels:
                continue
            labeled += 1
            judgment = result.metrics[name]
            if judgment.status == "skipped":
                skipped += 1
            elif judgment.status == "error":
                errors += 1
            else:
                # Evaluator's successful judgments always contain both scores.
                assert judgment.raw_score is not None and judgment.score is not None
                labels.append(row.labels[name])
                raw.append(judgment.raw_score)
                effective.append(judgment.score)
        evaluated = len(labels)
        raw_kind = metric.raw_kind
        metrics[name] = MetricValidationReport(
            labeled=labeled,
            evaluated=evaluated,
            unavailable=skipped + errors,
            skipped=skipped,
            errors=errors,
            coverage=evaluated / labeled if labeled else None,
            threshold=metric.threshold,
            raw_kind=raw_kind,
            effective_kind="event_probability" if evaluation.calibrated else raw_kind,
            raw=probability_diagnostics(raw, labels, threshold=metric.threshold)
            if evaluated
            else None,
            effective=probability_diagnostics(effective, labels, threshold=metric.threshold)
            if evaluated
            else None,
            raw_roc_auc=_roc_auc(raw, labels),
            effective_roc_auc=_roc_auc(effective, labels),
            majority_baseline_accuracy=max(sum(labels), evaluated - sum(labels)) / evaluated
            if evaluated
            else None,
        )
    return ValidationReport(
        examples=len(examples), calibrated=evaluation.calibrated, metrics=metrics
    )


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Probability a random positive outranks a negative, with half credit for ties."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    wins = 0.0
    previous_negatives = 0
    for _, tied in groupby(sorted(zip(scores, labels, strict=True)), key=lambda pair: pair[0]):
        group = tuple(tied)
        group_positives = sum(label for _, label in group)
        group_negatives = len(group) - group_positives
        wins += group_positives * (previous_negatives + 0.5 * group_negatives)
        previous_negatives += group_negatives
    return wins / (positives * negatives)
