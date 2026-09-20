import json

import pytest
from pydantic import ValidationError

from typed_evals import (
    AnswerRelevancy,
    EvaluationPipeline,
    EvaluationReport,
    Evaluator,
    Faithfulness,
    SampleResult,
    aevaluate,
    evaluate,
)
from typed_evals.errors import MissingInputError


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("form", ["fields", "sample", "dict", "batch", "mixed", "json", "jsonl"])
async def test_convenience_inputs_preserve_results_and_batch_shape(
    sample, backend, tmp_path, form, asynchronous
):
    data = sample.model_dump(mode="json")
    args, kwargs = (), {"backend": backend, "preset": "rag"}
    if form == "fields":
        kwargs.update(data)
    elif form in ("json", "jsonl"):
        path = tmp_path / f"samples.{form}"
        path.write_text(json.dumps([data] if form == "json" else data))
        args = (path if form == "json" else str(path),)
    else:
        args = ({"sample": sample, "dict": data, "batch": [data], "mixed": [sample, data]}[form],)
    result = await aevaluate(*args, **kwargs) if asynchronous else evaluate(*args, **kwargs)
    if form in ("fields", "sample", "dict"):
        assert isinstance(result, SampleResult)
        results = [result]
    else:
        assert isinstance(result, EvaluationReport)
        results = result.results
    assert len(results) == (2 if form == "mixed" else 1)
    assert all(row.passed and row.sample_id == sample.id for row in results)
    assert all(
        set(row.metrics) == {"faithfulness", "answer_relevancy", "context_relevance"}
        for row in results
    )
    assert backend.sessions == backend.closed == 1
    assert len(backend.calls) == len(results)


def test_existing_positional_metrics_and_empty_batch_are_compatible(sample, backend):
    report = evaluate([sample], [Faithfulness()], backend=backend)
    assert isinstance(report, EvaluationReport)
    assert set(report.summary) == {"faithfulness"}
    empty = evaluate([], backend=backend)
    assert isinstance(empty, EvaluationReport)
    assert empty.results == ()
    assert backend.sessions == 1


def test_default_does_not_infer_checks_from_extra_evidence(sample, backend):
    result = evaluate(sample, backend=backend)
    assert set(result.metrics) == {"answer_relevancy"}
    assert backend.calls[0][0] == {"input": sample.input, "response": sample.response}


@pytest.mark.parametrize(
    "preset,names",
    [
        ("response", ["answer_relevancy"]),
        ("rag", ["faithfulness", "answer_relevancy", "context_relevance"]),
        ("agent", ["task_completion", "tool_grounding"]),
    ],
)
def test_presets_work_in_reusable_evaluators_and_pipelines(preset, names, backend):
    evaluator = Evaluator(preset=preset, backend=backend)
    pipeline = EvaluationPipeline(preset=preset, backend=backend)
    assert [metric.name for metric in evaluator.metrics] == names
    assert evaluator.metrics == pipeline.evaluator.metrics
    assert all(
        a is not b for a, b in zip(evaluator.metrics, pipeline.evaluator.metrics, strict=True)
    )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"preset": "typo", "input": "Q", "response": "A"}, ValueError),
        ({"preset": "rag", "metrics": [AnswerRelevancy()]}, ValueError),
        ({"input": "Q", "response": "A", "respnse": "typo"}, ValidationError),
        ({"input": "Q"}, ValidationError),
        ({"input": "Q", "response": "A", "preset": "rag"}, MissingInputError),
        ({"samples": [], "input": "Q", "response": "A"}, TypeError),
        ({"samples": b"data"}, TypeError),
        ({"samples": [42]}, TypeError),
    ],
)
def test_bad_convenience_inputs_never_open_a_session(kwargs, error, backend):
    with pytest.raises(error):
        evaluate(**kwargs, backend=backend)
    assert backend.sessions == 0


def test_dictionary_batch_validates_every_row_before_judging(sample, backend):
    with pytest.raises(MissingInputError):
        evaluate([sample, {"input": "Q", "response": "A"}], preset="rag", backend=backend)
    with pytest.raises(ValidationError):
        evaluate([sample, {"input": "Q"}], backend=backend)
    assert backend.sessions == 0


def test_custom_metrics_and_explicit_partial_reports(sample, backend):
    report = evaluate(
        [sample, {"input": "Q", "response": "A"}],
        metrics=[Faithfulness(threshold=0.9)],
        missing="skip",
        backend=backend,
    )
    assert report.results[0].passed is False
    assert report.results[1].passed is None
    assert report.summary["faithfulness"].skipped == 1


async def test_async_convenience_preserves_concurrency_and_order(backend):
    backend.delay = 0.001
    report = await aevaluate(
        [{"input": f"Q{i}", "response": "A", "id": str(i)} for i in range(8)],
        backend=backend,
        max_concurrency=2,
    )
    assert backend.peak == 2
    assert [row.sample_id for row in report.results] == [str(i) for i in range(8)]
