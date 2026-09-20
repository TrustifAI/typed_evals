from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, overload

from pydantic import JsonValue

from typed_evals._utils import run_sync
from typed_evals.backends import Backend, JevBackend, JudgeSession, Question
from typed_evals.data.datasets import load_dataset
from typed_evals.data.models import (
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    MetricSummary,
    SampleResult,
    ToolCall,
    ToolProposal,
)
from typed_evals.errors import InvalidAnswerError, MissingInputError
from typed_evals.metrics import AnswerRelevancy, Metric
from typed_evals.metrics.base import _copy_metric
from typed_evals.metrics.presets import Preset, preset_metrics

if TYPE_CHECKING:
    from typed_evals.calibration import CalibrationBundle


class Evaluator:
    """Evaluate precomputed responses; this class never invokes the application under test."""

    def __init__(
        self,
        metrics: Sequence[Metric] | None = None,
        *,
        preset: Preset | None = None,
        backend: Backend | None = None,
        max_concurrency: int = 8,
        missing: Literal["raise", "skip"] = "raise",
        errors: Literal["raise", "record"] = "raise",
    ) -> None:
        if preset is not None and metrics is not None:
            raise ValueError("Choose preset or metrics, not both; use metrics for a custom panel")
        selected = (
            preset_metrics(preset)
            if preset is not None
            else ((AnswerRelevancy(),) if metrics is None else tuple(metrics))
        )
        if not selected:
            raise ValueError("metrics must contain at least one Metric")
        selected = tuple(_copy_metric(metric, index=index) for index, metric in enumerate(selected))
        if len({metric.name for metric in selected}) != len(selected):
            raise ValueError("metric names must be unique")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if missing not in ("raise", "skip") or errors not in ("raise", "record"):
            raise ValueError("invalid missing-input or error policy")
        # Own the definitions; subsequent edits to caller-owned nested criteria cannot alter them.
        self.metrics = selected
        self.backend = backend if backend is not None else JevBackend()
        self.max_concurrency, self.missing, self.errors = max_concurrency, missing, errors

    def evaluate(
        self,
        samples: Sequence[EvaluationSample],
        *,
        calibration: CalibrationBundle | None = None,
        allow_calibration_overlap: bool = False,
    ) -> EvaluationReport:
        return run_sync(
            lambda: self.aevaluate(
                samples,
                calibration=calibration,
                allow_calibration_overlap=allow_calibration_overlap,
            )
        )

    def evaluate_one(self, sample: EvaluationSample, **kwargs: Any) -> SampleResult:
        return self.evaluate([sample], **kwargs).results[0]

    async def aevaluate_one(self, sample: EvaluationSample, **kwargs: Any) -> SampleResult:
        return (await self.aevaluate([sample], **kwargs)).results[0]

    async def aevaluate(
        self,
        samples: Sequence[EvaluationSample],
        *,
        calibration: CalibrationBundle | None = None,
        allow_calibration_overlap: bool = False,
    ) -> EvaluationReport:
        samples = tuple(samples)
        if any(not isinstance(sample, EvaluationSample) for sample in samples):
            raise TypeError("Pass EvaluationSample objects; use load_dataset for JSON/JSONL files")
        if calibration is not None:
            calibration.validate_for(self.metrics, self.backend.model)
            if not allow_calibration_overlap:
                calibration.check_overlap(samples)
        questions = {metric.name: metric.question() for metric in self.metrics}
        prepared: list[tuple[dict[str, Any], dict[str, Question], dict[str, MetricResult]]] = []
        for sample in samples:
            active, skipped, fields = {}, {}, set()
            for metric in self.metrics:
                absent = metric.missing_fields(sample)
                if absent:
                    message = f"{metric.name}: missing {', '.join(absent)}"
                    if self.missing == "raise":
                        raise MissingInputError(message)
                    skipped[metric.name] = MetricResult(
                        name=metric.name,
                        status="skipped",
                        threshold=metric.threshold,
                        message=message,
                    )
                else:
                    active[metric.name] = questions[metric.name]
                    fields.update(metric.required_fields)
            prepared.append((sample.state(fields), active, skipped))
        results: list[SampleResult | None] = [None] * len(samples)
        pending = iter(range(len(samples)))

        async def worker(session: JudgeSession | None) -> None:
            for index in pending:
                # next() and index assignment happen without an await, so a worker owns its row.
                results[index] = await self._one(
                    samples[index],
                    index,
                    prepared[index],
                    session,
                    calibration,
                )

        if any(active for _, active, _ in prepared):
            async with self.backend.session() as session:
                tasks = [
                    asyncio.create_task(worker(session))
                    for _ in range(min(self.max_concurrency, len(samples)))
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await worker(None)
        completed = tuple(result for result in results if result is not None)
        summary = {}
        for metric in self.metrics:
            values = [result.metrics[metric.name] for result in completed]
            good = [value for value in values if value.status == "ok"]
            passed = sum(value.passed is True for value in good)
            summary[metric.name] = MetricSummary(
                evaluated=len(good),
                skipped=sum(value.status == "skipped" for value in values),
                errors=sum(value.status == "error" for value in values),
                passed=passed,
                mean_raw_score=sum(value.raw_score for value in good) / len(good) if good else None,
                mean_score=sum(value.score for value in good) / len(good) if good else None,
                pass_rate=passed / len(good) if good else None,
            )
        return EvaluationReport(
            results=completed, summary=summary, calibrated=calibration is not None
        )

    async def _one(
        self,
        sample: EvaluationSample,
        index: int,
        prepared: tuple,
        session: JudgeSession | None,
        calibration: CalibrationBundle | None,
    ) -> SampleResult:
        started = time.perf_counter()
        state, questions, skipped = prepared
        metrics, model, usage = dict(skipped), None, {}
        if questions:
            try:
                response = await session.judge(state, questions)
            except Exception as exc:
                if self.errors == "raise":
                    raise
                for metric in self.metrics:
                    if metric.name in questions:
                        metrics[metric.name] = self._error(metric, exc)
            else:
                model, usage = response.model, response.usage
                if calibration is not None:
                    # Configuration/model drift is fatal even in errors='record' mode.
                    calibration.check_model(model)
                for metric in self.metrics:
                    if metric.name not in questions:
                        continue
                    try:
                        if metric.name not in response.answers:
                            raise InvalidAnswerError(f"Jev omitted {metric.name}")
                        data = metric.read_answer(response.answers[metric.name])
                    except InvalidAnswerError as exc:
                        if self.errors == "raise":
                            raise
                        metrics[metric.name] = self._error(metric, exc)
                        continue
                    calibrated = (
                        calibration.curves[metric.name].predict(data["raw_score"])
                        if calibration is not None
                        else None
                    )
                    score = calibrated if calibrated is not None else data["raw_score"]
                    metrics[metric.name] = MetricResult(
                        name=metric.name,
                        status="ok",
                        threshold=metric.threshold,
                        passed=score >= metric.threshold,
                        calibrated_probability=calibrated,
                        calibration_target="metric_pass" if calibrated is not None else None,
                        **data,
                    )
        return SampleResult(
            sample_id=sample.id if sample.id is not None else str(index),
            tool_name=sample.proposed_tool_call.name
            if sample.proposed_tool_call is not None
            else None,
            sample_hash=sample.content_hash,
            model=model,
            metrics={metric.name: metrics[metric.name] for metric in self.metrics},
            usage=usage,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _error(metric: Metric, error: Exception) -> MetricResult:
        # Provider exception messages can contain request content. Reports store only the type.
        return MetricResult(
            name=metric.name,
            status="error",
            threshold=metric.threshold,
            message=f"Evaluation failed: {type(error).__name__}",
        )


class _EvaluationOptions(TypedDict, total=False):
    preset: Preset | None
    backend: Backend | None
    max_concurrency: int
    missing: Literal["raise", "skip"]
    errors: Literal["raise", "record"]


class _DirectOptions(_EvaluationOptions, total=False):
    input: str
    response: str
    contexts: Sequence[str]
    reference: str | None
    trace: Sequence[ToolCall | dict[str, Any]]
    expected_outcome: str | None
    proposed_tool_call: ToolProposal | dict[str, Any] | None
    id: str | None
    group_id: str | None
    metadata: dict[str, JsonValue]


SampleInput = EvaluationSample | Mapping[str, Any]
BatchInput = Sequence[SampleInput] | str | Path


def _prepare_inputs(
    samples: SampleInput | BatchInput | None, fields: dict[str, Any]
) -> tuple[tuple[EvaluationSample, ...], bool]:
    if samples is not None and fields:
        raise TypeError("Pass samples or direct sample fields, not both")
    if samples is None:
        return (EvaluationSample.model_validate(fields),), True
    if isinstance(samples, EvaluationSample):
        return (samples,), True
    if isinstance(samples, Mapping):
        return (EvaluationSample.model_validate(dict(samples)),), True
    if isinstance(samples, (str, Path)):
        return tuple(load_dataset(samples)), False
    if not isinstance(samples, Sequence) or isinstance(samples, (bytes, bytearray)):
        raise TypeError("Pass a sample, a sequence of samples/dictionaries, or a JSON/JSONL path")
    rows = []
    for index, sample in enumerate(samples):
        if isinstance(sample, EvaluationSample):
            rows.append(sample)
        elif isinstance(sample, Mapping):
            rows.append(EvaluationSample.model_validate(dict(sample)))
        else:
            raise TypeError(f"samples[{index}] must be an EvaluationSample or a dictionary")
    return tuple(rows), False


@overload
def evaluate(
    samples: SampleInput,
    metrics: Sequence[Metric] | None = None,
    **options: Unpack[_EvaluationOptions],
) -> SampleResult: ...


@overload
def evaluate(
    samples: BatchInput,
    metrics: Sequence[Metric] | None = None,
    **options: Unpack[_EvaluationOptions],
) -> EvaluationReport: ...


@overload
def evaluate(
    samples: None = None,
    metrics: Sequence[Metric] | None = None,
    **options: Unpack[_DirectOptions],
) -> SampleResult: ...


def evaluate(
    samples: SampleInput | BatchInput | None = None,
    metrics: Sequence[Metric] | None = None,
    *,
    preset: Preset | None = None,
    backend: Backend | None = None,
    max_concurrency: int = 8,
    missing: Literal["raise", "skip"] = "raise",
    errors: Literal["raise", "record"] = "raise",
    **sample_fields: Any,
) -> SampleResult | EvaluationReport:
    """Evaluate direct fields, one sample/dictionary, a batch, or a JSON/JSONL path.

    One sample returns SampleResult; sequences and paths always return EvaluationReport.
    Without metrics or a preset, checks answer relevancy only. Calibration stays opt-in
    through EvaluationPipeline. Use aevaluate to keep a running event loop responsive.
    """
    return run_sync(
        lambda: aevaluate(
            samples,
            metrics,
            preset=preset,
            backend=backend,
            max_concurrency=max_concurrency,
            missing=missing,
            errors=errors,
            **sample_fields,
        )
    )


@overload
async def aevaluate(
    samples: SampleInput,
    metrics: Sequence[Metric] | None = None,
    **options: Unpack[_EvaluationOptions],
) -> SampleResult: ...


@overload
async def aevaluate(
    samples: BatchInput,
    metrics: Sequence[Metric] | None = None,
    **options: Unpack[_EvaluationOptions],
) -> EvaluationReport: ...


@overload
async def aevaluate(
    samples: None = None,
    metrics: Sequence[Metric] | None = None,
    **options: Unpack[_DirectOptions],
) -> SampleResult: ...


async def aevaluate(
    samples: SampleInput | BatchInput | None = None,
    metrics: Sequence[Metric] | None = None,
    *,
    preset: Preset | None = None,
    backend: Backend | None = None,
    max_concurrency: int = 8,
    missing: Literal["raise", "skip"] = "raise",
    errors: Literal["raise", "record"] = "raise",
    **sample_fields: Any,
) -> SampleResult | EvaluationReport:
    """Async counterpart of evaluate, with identical inputs and result types."""
    evaluator = Evaluator(
        metrics,
        preset=preset,
        backend=backend,
        max_concurrency=max_concurrency,
        missing=missing,
        errors=errors,
    )
    rows, single = _prepare_inputs(samples, sample_fields)
    report = await evaluator.aevaluate(rows)
    return report.results[0] if single else report
