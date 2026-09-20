from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typed_evals._utils import digest, run_sync
from typed_evals.backends import Backend
from typed_evals.calibration import (
    CalibrationBundle,
    CalibrationConfig,
    CalibrationReport,
    IsotonicCalibrator,
    MetricCalibrationReport,
    probability_diagnostics,
)
from typed_evals.data.models import (
    CalibrationExample,
    EvaluationReport,
    EvaluationSample,
    SampleResult,
)
from typed_evals.errors import CalibrationError, DataLeakageError
from typed_evals.evaluation.evaluator import Evaluator
from typed_evals.metrics import Metric


class EvaluationPipeline:
    """Validate → judge → optionally fit/apply calibration → threshold → report."""

    def __init__(
        self,
        metrics: Sequence[Metric] | None = None,
        *,
        backend: Backend | None = None,
        calibration: CalibrationConfig | None = None,
        **evaluator_options: Any,
    ) -> None:
        self.evaluator = Evaluator(metrics, backend=backend, **evaluator_options)
        self.config = calibration if calibration is not None else CalibrationConfig()
        self._bundle: CalibrationBundle | None = None
        self._fitting = False

    @property
    def calibration_bundle(self) -> CalibrationBundle | None:
        return self._bundle

    @property
    def calibration_report(self) -> CalibrationReport | None:
        return self._bundle.report if self._bundle is not None else None

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise CalibrationError("Calibration is opt-in: pass CalibrationConfig(enabled=True)")

    def fit(
        self,
        data: Sequence[CalibrationExample],
        *,
        validation_data: Sequence[CalibrationExample] | None = None,
    ) -> CalibrationReport:
        return run_sync(lambda: self.afit(data, validation_data=validation_data))

    async def afit(
        self,
        data: Sequence[CalibrationExample],
        *,
        validation_data: Sequence[CalibrationExample] | None = None,
    ) -> CalibrationReport:
        self._require_enabled()
        if self._fitting:
            raise CalibrationError("This pipeline is already fitting calibration")
        self._fitting = True
        try:
            return await self._fit(
                tuple(data), None if validation_data is None else tuple(validation_data)
            )
        finally:
            self._fitting = False

    async def _fit(
        self,
        data: tuple[CalibrationExample, ...],
        validation_data: tuple[CalibrationExample, ...] | None,
    ) -> CalibrationReport:
        if not data or any(not isinstance(row, CalibrationExample) for row in data):
            raise CalibrationError("data must be a nonempty sequence of CalibrationExample objects")
        if validation_data is not None and (
            not validation_data
            or any(not isinstance(row, CalibrationExample) for row in validation_data)
        ):
            raise CalibrationError(
                "validation_data must be a nonempty sequence of CalibrationExample objects"
            )
        all_rows = data + (validation_data if validation_data is not None else ())
        evidence_fields = tuple(
            sorted({field for metric in self.evaluator.metrics for field in metric.required_fields})
        )
        hashes = [digest(row.sample.state(evidence_fields)) for row in all_rows]
        if len(set(hashes)) != len(hashes):
            raise DataLeakageError(
                "Calibration data contains repeated sample content; deduplicate first"
            )
        metric_map = {metric.name: metric for metric in self.evaluator.metrics}
        for row in all_rows:
            unknown = set(row.labels) - metric_map.keys()
            if unknown:
                raise CalibrationError(f"Unknown metric labels: {sorted(unknown)}")
            for name in row.labels:
                if metric_map[name].missing_fields(row.sample):
                    raise CalibrationError(f"Labeled metric {name} lacks required sample evidence")
        if validation_data is None:
            train, validation = self._split(data)
            split = "automatic_group_holdout"
        else:
            train, validation = data, validation_data
            split = "explicit_holdout"
            train_groups = {
                row.sample.group_hash for row in train if row.sample.group_hash is not None
            }
            validation_groups = {
                row.sample.group_hash for row in validation if row.sample.group_hash is not None
            }
            if train_groups & validation_groups:
                raise DataLeakageError("Training and validation group_id values overlap")
        for name in metric_map:
            self._validate_counts(name, train, training=True)
            self._validate_counts(name, validation, training=False)
        # Fail before making billable requests if fitting dependencies are unavailable.
        try:
            from sklearn.isotonic import IsotonicRegression  # noqa: F401
        except ImportError as exc:
            raise CalibrationError(
                "Install fitting dependencies with pip install 'typed_evals[calibration]'"
            ) from exc
        judge = Evaluator(
            self.evaluator.metrics,
            backend=self.evaluator.backend,
            max_concurrency=self.evaluator.max_concurrency,
            missing="skip",
            errors="raise",
        )
        # One shared connection pool and one request per sample, never one per metric.
        raw = await judge.aevaluate([row.sample for row in (*train, *validation)])
        actual_models = {result.model for result in raw.results}
        if len(actual_models) != 1 or None in actual_models:
            raise CalibrationError("Calibration must use one observed Jev model version")
        curves, reports = {}, {}
        for name in metric_map:
            x_train, y_train = self._xy(name, train, raw.results[: len(train)])
            x_val, y_val = self._xy(name, validation, raw.results[len(train) :])
            curve = IsotonicCalibrator.fit(x_train, y_train)
            before = probability_diagnostics(x_val, y_val, n_bins=self.config.n_bins)
            after = probability_diagnostics(
                [curve.predict(value) for value in x_val], y_val, n_bins=self.config.n_bins
            )
            notes = []
            if len(x_train) < 100:
                notes.append(
                    "Small fit set: this curve may be unstable. Validate with more representative labels."
                )
            if len(set(x_train)) == 1:
                notes.append(
                    "The raw signal is constant; calibration can learn only the training base rate."
                )
            if after.brier >= before.brier:
                notes.append(
                    "Held-out Brier score did not improve; inspect this metric before deployment."
                )
            curves[name] = curve
            reports[name] = MetricCalibrationReport(
                training_samples=len(x_train),
                validation_samples=len(x_val),
                training_positive_rate=sum(y_train) / len(y_train),
                raw=before,
                calibrated=after,
                brier_improved=after.brier < before.brier,
                notes=tuple(notes),
            )
        report = CalibrationReport(
            split=split, random_state=self.config.random_state, metrics=reports
        )
        bundle = CalibrationBundle(
            created_at=datetime.now(UTC).isoformat(),
            requested_model=self.evaluator.backend.model,
            observed_model=next(iter(actual_models)),
            metric_fingerprints={
                metric.name: metric.fingerprint for metric in self.evaluator.metrics
            },
            metric_order=tuple(metric_map),
            evidence_fields=evidence_fields,
            curves=curves,
            report=report,
            reserved_sample_hashes=frozenset(hashes),
            reserved_group_hashes=frozenset(
                row.sample.group_hash for row in all_rows if row.sample.group_hash is not None
            ),
        )
        # Publish only after all fits succeed. A failed refit cannot replace a working bundle.
        self._bundle = bundle
        return report

    def _split(self, data: tuple[CalibrationExample, ...]) -> tuple[tuple, tuple]:
        groups: dict[str, list[CalibrationExample]] = {}
        for row in data:
            key = row.sample.group_hash or row.sample.content_hash
            groups.setdefault(key, []).append(row)
        keys = list(groups)
        if len(keys) < 2:
            raise CalibrationError("Need at least two independent groups for a hold-out split")
        random.Random(self.config.random_state).shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * self.config.validation_fraction)))
        validation_keys = set(keys[:count])
        train = tuple(
            row for key, rows in groups.items() if key not in validation_keys for row in rows
        )
        validation = tuple(
            row for key, rows in groups.items() if key in validation_keys for row in rows
        )
        return train, validation

    def _validate_counts(
        self, name: str, rows: Sequence[CalibrationExample], *, training: bool
    ) -> None:
        labels = [row.labels[name] for row in rows if name in row.labels]
        minimum = self.config.min_samples if training else self.config.min_validation_samples
        side = "training" if training else "validation"
        if len(labels) < minimum:
            raise CalibrationError(
                f"{name}: {side} needs at least {minimum} labeled examples; got {len(labels)}"
            )
        class_minimum = self.config.min_class_samples if training else 1
        if min(labels.count(0), labels.count(1)) < class_minimum:
            raise CalibrationError(
                f"{name}: {side} needs at least {class_minimum} examples of each class; "
                "supply a balanced explicit hold-out if the automatic split is unsuitable"
            )

    @staticmethod
    def _xy(
        name: str, rows: Sequence[CalibrationExample], results: Sequence[SampleResult]
    ) -> tuple[list[float], list[int]]:
        x, y = [], []
        for row, result in zip(rows, results, strict=True):
            if name not in row.labels:
                continue
            metric = result.metrics[name]
            if metric.status != "ok" or metric.raw_score is None:
                raise CalibrationError(f"{name}: labeled calibration sample was not evaluated")
            x.append(metric.raw_score)
            y.append(row.labels[name])
        return x, y

    def _active_bundle(self) -> CalibrationBundle | None:
        if self._fitting:
            raise CalibrationError("Wait for calibration fitting to finish before evaluation")
        if not self.config.enabled:
            return None
        if self._bundle is None:
            raise CalibrationError(
                "Calibration is enabled but unfitted; call fit, run with data, or load_calibration"
            )
        return self._bundle

    def evaluate(
        self, samples: Sequence[EvaluationSample], *, allow_calibration_overlap: bool = False
    ) -> EvaluationReport:
        return run_sync(
            lambda: self.aevaluate(samples, allow_calibration_overlap=allow_calibration_overlap)
        )

    async def aevaluate(
        self, samples: Sequence[EvaluationSample], *, allow_calibration_overlap: bool = False
    ) -> EvaluationReport:
        return await self.evaluator.aevaluate(
            samples,
            calibration=self._active_bundle(),
            allow_calibration_overlap=allow_calibration_overlap,
        )

    def evaluate_one(self, sample: EvaluationSample, **kwargs: Any) -> SampleResult:
        return self.evaluate([sample], **kwargs).results[0]

    async def aevaluate_one(self, sample: EvaluationSample, **kwargs: Any) -> SampleResult:
        return (await self.aevaluate([sample], **kwargs)).results[0]

    def run(
        self,
        samples: Sequence[EvaluationSample],
        *,
        calibration_data: Sequence[CalibrationExample] | None = None,
        validation_data: Sequence[CalibrationExample] | None = None,
    ) -> EvaluationReport:
        return run_sync(
            lambda: self.arun(
                samples, calibration_data=calibration_data, validation_data=validation_data
            )
        )

    async def arun(
        self,
        samples: Sequence[EvaluationSample],
        *,
        calibration_data: Sequence[CalibrationExample] | None = None,
        validation_data: Sequence[CalibrationExample] | None = None,
    ) -> EvaluationReport:
        if validation_data is not None and calibration_data is None:
            raise CalibrationError("validation_data requires calibration_data")
        if calibration_data is not None:
            await self.afit(calibration_data, validation_data=validation_data)
        return await self.aevaluate(samples)

    def save_calibration(self, path: str | Path) -> None:
        self._require_enabled()
        bundle = self._active_bundle()
        bundle.save(path)

    def load_calibration(self, path: str | Path) -> EvaluationPipeline:
        self._require_enabled()
        if self._fitting:
            raise CalibrationError("Cannot load calibration while a fit is running")
        bundle = CalibrationBundle.load(path)
        bundle.validate_for(self.evaluator.metrics, self.evaluator.backend.model)
        self._bundle = bundle
        return self
