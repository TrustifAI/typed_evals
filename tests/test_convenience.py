import json
import warnings

import pytest
from pydantic import ValidationError

from typed_evals import (
    AnswerRelevancy,
    EvaluationPipeline,
    EvaluationReport,
    EvaluationSample,
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
    second = sample.model_copy(update={"id": "second", "input": "Different Q", "response": "Wrong"})
    backend.answer = lambda state, questions: {
        name: {"type": "noul", "noul": 0.2 if state["input"] == "Different Q" else 0.8}
        for name in questions
    }
    args, kwargs = (), {"backend": backend, "preset": "rag"}
    if form == "fields":
        kwargs.update(data)
    elif form in ("json", "jsonl"):
        path = tmp_path / f"samples.{form}"
        path.write_text(json.dumps([data] if form == "json" else data))
        args = (path if form == "json" else str(path),)
    else:
        args = (
            {
                "sample": sample,
                "dict": data,
                "batch": [data],
                "mixed": [sample, second.model_dump(mode="json")],
            }[form],
        )
    result = await aevaluate(*args, **kwargs) if asynchronous else evaluate(*args, **kwargs)
    if form in ("fields", "sample", "dict"):
        assert isinstance(result, SampleResult)
        results = [result]
    else:
        assert isinstance(result, EvaluationReport)
        results = result.results
    expected = [(sample.id, True, 0.8)]
    if form == "mixed":
        expected.append(("second", False, 0.2))
    assert [
        (row.sample_id, row.passed, row.metrics["answer_relevancy"].raw_score) for row in results
    ] == expected
    assert all(
        set(row.metrics)
        == {
            "faithfulness",
            "answer_relevancy",
            "answer_completeness",
            "context_relevance",
            "context_sufficiency",
        }
        for row in results
    )
    assert backend.sessions == backend.closed == 1
    assert len(backend.calls) == 2 * len(results)
    assert [state["response"] for state, _ in backend.calls if "response" in state] == (
        [sample.response, "Wrong"] if form == "mixed" else [sample.response]
    )


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
    "preset,names,fields",
    [
        ("response", ["answer_relevancy", "answer_completeness"], {"input", "response"}),
        (
            "rag",
            [
                "faithfulness",
                "answer_relevancy",
                "answer_completeness",
                "context_relevance",
                "context_sufficiency",
            ],
            {"input", "response", "contexts"},
        ),
        (
            "agent",
            ["task_completion", "tool_grounding", "answer_relevancy", "answer_completeness"],
            {"input", "response", "trace", "expected_outcome"},
        ),
    ],
)
@pytest.mark.parametrize("evaluator_type", [Evaluator, EvaluationPipeline])
def test_presets_evaluate_the_selected_checks_using_only_their_evidence(
    preset, names, fields, evaluator_type, backend
):
    sample = EvaluationSample(
        input="Look up order 42",
        response="The order was delivered.",
        contexts=("Delivery records",) if preset == "rag" else (),
        trace=({"name": "lookup", "arguments": {"id": 42}, "output": "Delivered"},),
        expected_outcome="Find the delivery status",
        reference="private reference unused by these panels",
        metadata={"label": "private label"},
    )
    scores = {name: index / len(names) for index, name in enumerate(names, 1)}
    backend.answer = lambda state, questions: {
        name: {
            "type": "noul",
            "noul": scores["context_relevance"]
            if "target_context_index" in question.instructions
            else scores[name],
        }
        for name, question in questions.items()
    }
    result = evaluator_type(preset=preset, backend=backend).evaluate_one(sample)

    assert list(result.metrics) == names
    expected_scores = dict(scores)
    if preset == "rag":
        expected_scores["context_relevance"] = float(scores["context_relevance"] >= 0.5)
    assert {name: metric.raw_score for name, metric in result.metrics.items()} == expected_scores
    assert result.passed is (min(scores.values()) >= 0.5)
    assert len(backend.calls) == (2 if preset == "rag" else 1)
    state, questions = backend.calls[0]
    assert set(state) == fields
    assert list(questions) == [name for name in names if name != "context_relevance"]
    if preset == "agent":
        assert state["trace"][0]["output"] == "Delivered"
        assert state["expected_outcome"] == "Find the delivery status"
    assert backend.sessions == backend.closed == 1


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
