import json
import warnings

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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("form", ["fields", "sample", "dict", "batch", "jsonl"])
async def test_default_warns_without_inferring_checks_from_contexts(
    sample, backend, tmp_path, form, asynchronous
):
    data = sample.model_dump(mode="json")
    args, kwargs = (), {"backend": backend}
    if form == "fields":
        kwargs.update(data)
    elif form == "jsonl":
        path = tmp_path / "samples.jsonl"
        path.write_text(json.dumps(data) + "\n")
        args = (path,)
    else:
        args = ({"sample": sample, "dict": data, "batch": [sample]}[form],)
    with pytest.warns(UserWarning, match="contexts were supplied for 1 sample") as caught:
        output = await aevaluate(*args, **kwargs) if asynchronous else evaluate(*args, **kwargs)
    assert len(caught) == 1
    result = output.results[0] if isinstance(output, EvaluationReport) else output
    assert set(result.metrics) == {"answer_relevancy"}
    assert backend.calls[0][0] == {"input": sample.input, "response": sample.response}


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_relevant_but_unsupported_answer_warns_and_needs_grounding_check(
    backend, asynchronous
):
    backend.answer = lambda state, questions: {
        name: {"type": "noul", "noul": 0.1 if name == "faithfulness" else 0.9} for name in questions
    }
    fields = {
        "input": "What is the refund period?",
        "response": "90 days.",
        "contexts": ["Refunds within 30 days."],
        "backend": backend,
    }
    with pytest.warns(UserWarning, match="contexts were supplied"):
        default = await aevaluate(**fields) if asynchronous else evaluate(**fields)
    assert default.passed is True
    assert set(default.metrics) == {"answer_relevancy"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        grounded = (
            await aevaluate(**fields, preset="rag")
            if asynchronous
            else evaluate(**fields, preset="rag")
        )
    assert grounded.passed is False
    assert grounded.metrics["answer_relevancy"].passed is True
    assert grounded.metrics["faithfulness"].passed is False
    assert backend.calls[1][0]["contexts"] == fields["contexts"]


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
