import builtins
import json

import pytest

from typed_evals import (
    EvaluationPipeline,
    EvaluationReport,
    EvaluationSample,
    Evaluator,
    IsotonicCalibrator,
    ToolAccuracy,
    ToolProposal,
    evaluated_by,
    load_calibration_dataset,
    load_dataset,
)
from typed_evals.cli import main
from typed_evals.data.models import SampleResult
from typed_evals.errors import CalibrationError, MissingInputError


def test_json_and_jsonl_datasets(tmp_path, sample):
    data = sample.model_dump(mode="json")
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([data]))
    assert load_dataset(path) == [sample]
    path = tmp_path / "samples.jsonl"
    path.write_text("\n" + json.dumps(data) + "\n\n")
    assert load_dataset(path) == [sample]
    path = tmp_path / "calibration.jsonl"
    path.write_text(json.dumps({"sample": data, "labels": {"faithfulness": True}}) + "\n")
    assert load_calibration_dataset(path)[0].labels == {"faithfulness": 1}


@pytest.mark.parametrize(
    "suffix,payload",
    [
        ("json", "{}"),
        ("jsonl", "{broken}"),
        ("json", '[{"input": "Q", "response": "A", "typo": 1}]'),
        ("txt", "anything"),
    ],
)
def test_bad_datasets_have_actionable_errors(tmp_path, suffix, payload):
    path = tmp_path / f"data.{suffix}"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_sync_decorator_preserves_native_output(backend):
    native = {"answer": "Paris", "sources": ["France"]}

    @evaluated_by(
        Evaluator(backend=backend),
        sample_builder=lambda output, args, kwargs: EvaluationSample(
            input=args[0], response=output["answer"]
        ),
    )
    def rag(question):
        return native

    result = rag("What is France's capital?")
    assert result.output is native
    assert isinstance(result.evaluation, SampleResult)
    assert result.evaluation.metrics["answer_relevancy"].score == 0.8
    assert rag.__name__ == "rag"


async def test_async_decorator_evaluates_awaited_result(backend):
    @evaluated_by(
        Evaluator(backend=backend),
        sample_builder=lambda output, args, kwargs: EvaluationSample(
            input=kwargs["question"], response=output
        ),
    )
    async def agent(*, question):
        return "Done"

    result = await agent(question="Run a task")
    assert result.output == "Done"
    assert isinstance(result.evaluation, SampleResult)
    assert result.evaluation.passed


@pytest.mark.parametrize("async_function", [False, True])
@pytest.mark.parametrize("evaluator_type", [Evaluator, EvaluationPipeline])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("count", [0, 1, 3])
async def test_decorator_evaluates_tool_call_batches(
    backend, async_function, evaluator_type, container, count
):
    native = {"answer": "Done"}
    tool_names = ["addition_tool", "subtraction_tool", "addition_tool"][:count]
    builder_calls = []

    def build_samples(output, args, kwargs):
        builder_calls.append((output, args, kwargs))
        return container(
            EvaluationSample(
                input=args[0] if args else kwargs["question"],
                response=output["answer"],
                contexts=(
                    "addition_tool(a, b) adds integers; subtraction_tool(a, b) subtracts them.",
                ),
                proposed_tool_call=ToolProposal(name=name, arguments={"a": index, "b": 1}),
            )
            for index, name in enumerate(tool_names)
        )

    def agent(question):
        return native

    async def async_agent(*, question):
        return native

    function = async_agent if async_function else agent
    wrapped = evaluated_by(
        evaluator_type([ToolAccuracy()], backend=backend), sample_builder=build_samples
    )(function)
    result = await wrapped(question="Calculate") if async_function else wrapped("Calculate")

    assert result.output is native
    assert wrapped.__name__ == function.__name__
    assert len(builder_calls) == 1
    assert builder_calls[0][0] is native
    report = result.evaluation
    assert isinstance(report, EvaluationReport)
    assert [row.sample_id for row in report.results] == [str(index) for index in range(count)]
    assert [row.tool_name for row in report.results] == tool_names
    assert all(row.metrics["tool_accuracy"].score == 0.8 for row in report.results)
    assert report.summary["tool_accuracy"].evaluated == count
    assert report.summary["tool_accuracy"].passed == count
    assert len(report.model_dump(mode="json")["results"]) == count
    assert len(backend.calls) == count
    assert all(state["input"] == "Calculate" for state, _ in backend.calls)
    assert [state["proposed_tool_call"]["arguments"] for state, _ in backend.calls] == [
        {"a": index, "b": 1} for index in range(count)
    ]
    assert backend.sessions == backend.closed == int(count > 0)


@pytest.mark.parametrize("async_function", [False, True])
@pytest.mark.parametrize("invalid_item", [False, True])
async def test_decorator_validates_entire_batch_before_judging(
    backend, async_function, invalid_item
):
    valid = EvaluationSample(
        input="Add 1 and 2",
        response="3",
        contexts=("addition_tool(a, b) adds two integers.",),
        proposed_tool_call=ToolProposal(name="addition_tool", arguments={"a": 1, "b": 2}),
    )
    invalid = (
        {"input": "Q", "response": "A"}
        if invalid_item
        else EvaluationSample(input="Q", response="A")
    )

    def agent():
        return "Done"

    async def async_agent():
        return "Done"

    wrapped = evaluated_by(
        Evaluator([ToolAccuracy()], backend=backend),
        sample_builder=lambda *args: [valid, invalid],
    )(async_agent if async_function else agent)
    with pytest.raises(TypeError if invalid_item else MissingInputError):
        if async_function:
            await wrapped()
        else:
            wrapped()
    assert not backend.calls
    assert backend.sessions == 0


@pytest.mark.parametrize("async_function", [False, True])
async def test_decorator_propagates_batch_evaluation_errors(backend, sample, async_function):
    def fail(state, questions):
        raise RuntimeError("judge failed")

    backend.answer = fail

    def agent():
        return "Done"

    async def async_agent():
        return "Done"

    wrapped = evaluated_by(
        Evaluator(backend=backend), sample_builder=lambda *args: [sample, sample]
    )(async_agent if async_function else agent)
    with pytest.raises(RuntimeError, match="judge failed"):
        if async_function:
            await wrapped()
        else:
            wrapped()
    assert backend.sessions == backend.closed == 1


def test_generator_decorator_requires_explicit_materialization(backend):
    def stream():
        yield "piece"

    with pytest.raises(TypeError, match="materialized"):
        evaluated_by(Evaluator(backend=backend), sample_builder=lambda *args: None)(stream)


def test_sklearn_is_optional_for_inference_but_required_for_fit(monkeypatch, backend, sample):
    curve = IsotonicCalibrator(x=[0.2, 0.8], y=[0.3, 0.7], n_samples=100)
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    assert curve.predict(0.5) == pytest.approx(0.5)
    assert Evaluator(backend=backend).evaluate_one(sample).passed
    with pytest.raises(CalibrationError, match=r"typed_evals\[calibration\]"):
        IsotonicCalibrator.fit([0.1, 0.2, 0.7, 0.8], [0, 0, 1, 1])


def test_cli_writes_report_and_ci_exit_status(tmp_path, monkeypatch, backend, sample):
    import typed_evals.cli as cli

    monkeypatch.setattr(cli, "JevBackend", lambda **kwargs: backend)
    dataset = tmp_path / "data.json"
    output = tmp_path / "report.json"
    dataset.write_text(json.dumps([sample.model_dump(mode="json")]))
    assert main(["evaluate", str(dataset), "--output", str(output)]) == 0
    assert output.exists()
    assert (
        main(
            [
                "evaluate",
                str(dataset),
                "--output",
                str(output),
                "--threshold",
                "0.9",
                "--fail-on-failure",
            ]
        )
        == 1
    )


def test_cli_invalid_dataset_returns_error_without_network(tmp_path, monkeypatch, backend, capsys):
    import typed_evals.cli as cli

    monkeypatch.setattr(cli, "JevBackend", lambda **kwargs: backend)
    assert (
        main(["evaluate", str(tmp_path / "missing.json"), "--output", str(tmp_path / "x.json")])
        == 2
    )
    assert not backend.calls
    assert capsys.readouterr().err.startswith("typed_evals failed (FileNotFoundError).")
