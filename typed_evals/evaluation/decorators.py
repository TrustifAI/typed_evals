"""Framework-neutral wrappers. Map native outputs explicitly using a sample builder."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, Protocol, TypeVar

from typed_evals.data.models import EvaluationReport, EvaluationSample, SampleResult

T = TypeVar("T")


class ResponseEvaluator(Protocol):
    def evaluate_one(self, sample: EvaluationSample) -> SampleResult: ...

    async def aevaluate_one(self, sample: EvaluationSample) -> SampleResult: ...

    def evaluate(self, samples: Sequence[EvaluationSample]) -> EvaluationReport: ...

    async def aevaluate(self, samples: Sequence[EvaluationSample]) -> EvaluationReport: ...


@dataclass(frozen=True)
class EvaluatedResponse(Generic[T]):
    """Preserves the application's native output, alongside structured evaluation metadata."""

    output: T
    evaluation: SampleResult | EvaluationReport


def evaluated_by(
    evaluator: ResponseEvaluator,
    *,
    sample_builder: Callable[[Any, tuple, dict], EvaluationSample | Sequence[EvaluationSample]],
) -> Callable:
    """sample_builder(output, positional_args, keyword_args) must be a synchronous mapper.

    Return one EvaluationSample for a SampleResult, or a sequence of samples for an
    EvaluationReport. Empty sequences produce an empty report without judging any samples.
    Supports ordinary sync/async functions. Materialize streaming output before evaluation.
    Evaluation failures propagate; this prevents silently presenting an unevaluated result as trusted.
    """

    def decorator(function: Callable) -> Callable:
        if inspect.isasyncgenfunction(function) or inspect.isgeneratorfunction(function):
            raise TypeError("Streaming functions must be materialized before evaluation")
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> EvaluatedResponse:
                output = await function(*args, **kwargs)
                samples = sample_builder(output, args, kwargs)
                evaluation = (
                    await evaluator.aevaluate_one(samples)
                    if isinstance(samples, EvaluationSample)
                    else await evaluator.aevaluate(samples)
                )
                return EvaluatedResponse(output, evaluation)

            return async_wrapper

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> EvaluatedResponse:
            output = function(*args, **kwargs)
            samples = sample_builder(output, args, kwargs)
            evaluation = (
                evaluator.evaluate_one(samples)
                if isinstance(samples, EvaluationSample)
                else evaluator.evaluate(samples)
            )
            return EvaluatedResponse(output, evaluation)

        return wrapper

    return decorator
