from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast, overload

from pydantic import JsonValue

from typed_evals._utils import run_sync
from typed_evals.backends import Backend, JevBackend, JudgeResponse, JudgeSession, Question
from typed_evals.backends.capabilities import validate_evidence
from typed_evals.backends.provenance import backend_provenance
from typed_evals.data.datasets import load_dataset
from typed_evals.data.models import (
    ContextRelevanceDetails,
    ContextUnitResult,
    EvaluationReport,
    EvaluationSample,
    ImageInput,
    MetricResult,
    MetricSummary,
    SampleResult,
    ToolCall,
    ToolProposal,
)
from typed_evals.errors import (
    CalibrationMismatchError,
    DecisionRefusalError,
    InvalidAnswerError,
    MissingInputError,
)
from typed_evals.metrics import AnswerRelevancy, Metric
from typed_evals.metrics.base import _copy_metric
from typed_evals.metrics.context_units import (
    CONTEXT_SEGMENTATION_VERSION,
    ContextUnit,
    segment_contexts,
)
from typed_evals.metrics.presets import Preset, preset_metrics

if TYPE_CHECKING:
    from typed_evals.calibration import CalibrationBundle


@dataclass
class _ContextRequest:
    state: dict[str, Any]
    questions: dict[str, Question]
    units: tuple[ContextUnit, ...]


@dataclass
class _PreparedSample:
    state: dict[str, Any]
    questions: dict[str, Question]
    skipped: dict[str, MetricResult]
    context_metrics: tuple[tuple[Metric, tuple[_ContextRequest, ...]], ...] = ()

    @property
    def needs_judge(self) -> bool:
        return bool(self.questions or any(requests for _, requests in self.context_metrics))


@dataclass
class _ResponseSummary:
    """Combine actual responses without inventing missing usage or model identity."""

    models: set[str] = field(default_factory=set)
    model: str | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    responses: int = 0

    def observe(self, response: JudgeResponse, calibration: CalibrationBundle | None) -> None:
        if calibration is not None:
            # Configuration/model drift is fatal even in errors='record' mode.
            calibration.check_model(response.model)
        self.models.add(response.model)
        if not self.responses:
            self.model = response.model
            self.usage = dict(response.usage)
        else:
            merged = {}
            for key in self.usage.keys() | response.usage.keys():
                previous, incoming = self.usage.get(key), response.usage.get(key)
                merged[key] = (
                    previous + incoming if previous is not None and incoming is not None else None
                )
            self.usage = merged
        self.responses += 1


class Evaluator:
    """Evaluate precomputed responses; this class never invokes the application under test.

    Supplied contexts unused by active metrics produce one warning per batch.
    """

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
        questions = {metric.name: metric.question() for metric in self.metrics}
        if calibration is not None:
            calibration.validate_for(
                self.metrics,
                self.backend.model,
                backend_provenance=backend_provenance(self.backend, questions),
            )
            if not allow_calibration_overlap:
                calibration.check_overlap(samples)
        prepared: list[_PreparedSample] = []
        unused_contexts = 0
        for index, sample in enumerate(samples):
            active: dict[str, Question] = {}
            skipped: dict[str, MetricResult] = {}
            fields: set[str] = set()
            ordinary_fields: set[str] = set()
            context_metrics: list[tuple[Metric, tuple[_ContextRequest, ...]]] = []
            for metric in self.metrics:
                absent = metric.missing_fields(sample)
                if absent:
                    message = f"{metric.name}: missing {', '.join(absent)}"
                    if calibration is not None:
                        raise CalibrationMismatchError(
                            f"Calibrated sample {index}: {message}. Calibration requires the "
                            "complete metric panel on every sample, even with missing='skip'; "
                            "supply the required evidence or evaluate without calibration."
                        )
                    if self.missing == "raise":
                        raise MissingInputError(message)
                    skipped[metric.name] = MetricResult(
                        name=metric.name,
                        status="skipped",
                        threshold=metric.threshold,
                        message=message,
                    )
                else:
                    fields.update(metric.required_fields)
                    if metric.aggregation == "relevant_fraction":
                        requests = self._context_requests(sample, metric, questions[metric.name])
                        for request in requests:
                            validate_evidence(self.backend, request.state)
                        context_metrics.append((metric, requests))
                    else:
                        active[metric.name] = questions[metric.name]
                        ordinary_fields.update(metric.required_fields)
            state = sample.state(ordinary_fields)
            if active:
                validate_evidence(self.backend, state)
            if sample.contexts and "contexts" not in fields:
                unused_contexts += 1
            prepared.append(_PreparedSample(state, active, skipped, tuple(context_metrics)))
        if unused_contexts:
            warnings.warn(
                f"contexts were supplied for {unused_contexts} sample(s) but are unused because "
                "no active metric requires 'contexts'. Select preset='rag' or a metric that "
                "requires 'contexts'; skipped metrics do not consume evidence.",
                UserWarning,
                # The async entry point may be run by asyncio or a notebook worker thread;
                # keep the source here instead of attributing the warning to loop internals.
                stacklevel=1,
            )
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

        if any(item.needs_judge for item in prepared):
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
                mean_raw_score=sum(cast(float, value.raw_score) for value in good) / len(good)
                if good
                else None,
                mean_score=sum(cast(float, value.score) for value in good) / len(good)
                if good
                else None,
                pass_rate=passed / len(good) if good else None,
            )
        return EvaluationReport(
            results=completed, summary=summary, calibrated=calibration is not None
        )

    async def _one(
        self,
        sample: EvaluationSample,
        index: int,
        prepared: _PreparedSample,
        session: JudgeSession | None,
        calibration: CalibrationBundle | None,
    ) -> SampleResult:
        started = time.perf_counter()
        state, questions = prepared.state, prepared.questions
        metrics = dict(prepared.skipped)
        observed = _ResponseSummary()
        if questions:
            try:
                response = await cast(JudgeSession, session).judge(state, questions)
            except Exception as exc:
                if self.errors == "raise":
                    raise
                for metric in self.metrics:
                    if metric.name in questions:
                        metrics[metric.name] = self._error(metric, exc)
            else:
                observed.observe(response, calibration)
                for metric in self.metrics:
                    if metric.name not in questions:
                        continue
                    try:
                        data = self._read_answer(metric, response, metric.name)
                    except InvalidAnswerError as exc:
                        if self.errors == "raise":
                            raise
                        metrics[metric.name] = self._error(metric, exc)
                        continue
                    metrics[metric.name] = self._result(metric, data, calibration)
        for metric, requests in prepared.context_metrics:
            metrics[metric.name] = await self._context_result(
                metric, requests, session, calibration, observed
            )
        return SampleResult(
            sample_id=sample.id if sample.id is not None else str(index),
            tool_name=sample.proposed_tool_call.name
            if sample.proposed_tool_call is not None
            else None,
            sample_hash=sample.content_hash,
            model=observed.model,
            metrics={metric.name: metrics[metric.name] for metric in self.metrics},
            usage=observed.usage,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _context_requests(
        sample: EvaluationSample, metric: Metric, template: Question
    ) -> tuple[_ContextRequest, ...]:
        units = segment_contexts(sample.contexts or (), granularity=metric.granularity)
        requests = []
        for start in range(0, len(units), metric.questions_per_call):
            chunk = units[start : start + metric.questions_per_call]
            questions: dict[str, Question] = {}
            for context_index, unit in enumerate(chunk):
                instructions = deepcopy(dict(cast(Mapping[str, Any], template.instructions)))
                instructions.update(
                    target_context_index=context_index,
                    target_passage_index=str(unit.passage_index),
                )
                name = f"{metric.name}__p{unit.passage_index}_u{unit.unit_index}"
                questions[name] = template.model_copy(
                    deep=True, update={"instructions": instructions}
                )
            requests.append(
                _ContextRequest(
                    state={
                        "input": sample.input,
                        "contexts": [unit.text for unit in chunk],
                        # Passage context only disambiguates the target unit; it cannot
                        # make unrelated information in that unit relevant.
                        "context_passages": {
                            str(unit.passage_index): cast(tuple[str, ...], sample.contexts)[
                                unit.passage_index
                            ]
                            for unit in chunk
                        },
                    },
                    questions=questions,
                    units=chunk,
                )
            )
        return tuple(requests)

    async def _context_result(
        self,
        metric: Metric,
        requests: tuple[_ContextRequest, ...],
        session: JudgeSession | None,
        calibration: CalibrationBundle | None,
        observed: _ResponseSummary,
    ) -> MetricResult:
        units: list[ContextUnitResult] = []
        first_error: Exception | None = None
        for request in requests:
            try:
                response = await cast(JudgeSession, session).judge(request.state, request.questions)
            except Exception as exc:
                if self.errors == "raise":
                    raise
                first_error = first_error or exc
                units.extend(self._unit_error(unit, exc) for unit in request.units)
                continue
            observed.observe(response, calibration)
            if len(observed.models) > 1:
                error = InvalidAnswerError(
                    "Context unit judgments returned different model versions"
                )
                if self.errors == "raise":
                    raise error
                first_error = first_error or error
                units.extend(self._unit_error(unit, error) for unit in request.units)
                continue
            for name, unit in zip(request.questions, request.units, strict=True):
                try:
                    data = self._read_answer(metric, response, name)
                except InvalidAnswerError as exc:
                    if self.errors == "raise":
                        raise
                    first_error = first_error or exc
                    units.append(self._unit_error(unit, exc))
                else:
                    probability = cast(float, data["raw_score"])
                    units.append(
                        ContextUnitResult(
                            passage_index=unit.passage_index,
                            unit_index=unit.unit_index,
                            start=unit.start,
                            end=unit.end,
                            status="ok",
                            probability=probability,
                            relevant=probability >= metric.unit_threshold,
                        )
                    )
        total = len(units)
        evaluated = sum(unit.status == "ok" for unit in units)
        relevant = sum(unit.relevant is True for unit in units)
        details = ContextRelevanceDetails(
            granularity=metric.granularity,
            segmentation_version=CONTEXT_SEGMENTATION_VERSION,
            unit_threshold=metric.unit_threshold,
            total_units=total,
            evaluated_units=evaluated,
            relevant_units=relevant,
            unavailable_units=total - evaluated,
            coverage=evaluated / total if total else None,
            units=tuple(units),
        )
        if first_error is not None:
            return self._error(metric, first_error).model_copy(update={"details": details})
        return self._result(
            metric,
            {
                "raw_score": relevant / total if total else 0.0,
                "raw_kind": "relevance_fraction",
                "details": details,
            },
            calibration,
        )

    @staticmethod
    def _unit_error(unit: ContextUnit, error: Exception) -> ContextUnitResult:
        return ContextUnitResult(
            passage_index=unit.passage_index,
            unit_index=unit.unit_index,
            start=unit.start,
            end=unit.end,
            status="error",
            error_type=type(error).__name__,
        )

    @staticmethod
    def _read_answer(metric: Metric, response: JudgeResponse, name: str) -> dict[str, Any]:
        unavailable = getattr(response, "answer_errors", {}).get(name)
        if unavailable == "refusal":
            raise DecisionRefusalError(f"{name}: provider refused judgment")
        if unavailable is not None:
            raise InvalidAnswerError(f"{name}: invalid provider answer")
        if name not in response.answers:
            raise InvalidAnswerError(f"Judge omitted {name}")
        return metric.read_answer(response.answers[name])

    @staticmethod
    def _result(
        metric: Metric, data: dict[str, Any], calibration: CalibrationBundle | None
    ) -> MetricResult:
        calibrated = (
            calibration.curves[metric.name].predict(data["raw_score"])
            if calibration is not None
            else None
        )
        score = calibrated if calibrated is not None else data["raw_score"]
        return MetricResult(
            name=metric.name,
            status="ok",
            threshold=metric.threshold,
            passed=score >= metric.threshold,
            calibrated_probability=calibrated,
            calibration_target="metric_pass" if calibrated is not None else None,
            **data,
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
    contexts: Sequence[str] | None
    authorization: Sequence[str]
    images: Sequence[ImageInput | dict[str, Any]]
    reference: str | None
    trace: Sequence[ToolCall | dict[str, Any]] | None
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
    Warns once per batch when supplied contexts are unused by active metrics.
    """
    return cast(
        SampleResult | EvaluationReport,
        run_sync(
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
        ),
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
