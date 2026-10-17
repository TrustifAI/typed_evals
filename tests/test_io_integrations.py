import asyncio
import builtins
import inspect
import json
import traceback

import pytest
from conftest import calibrated_rows

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
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


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_json_and_jsonl_datasets(tmp_path, sample, suffix):
    data = sample.model_dump(mode="json")
    path = tmp_path / f"samples.{suffix}"
    path.write_text(json.dumps([data]) if suffix == "json" else "\n" + json.dumps(data) + "\n\n")
    assert load_dataset(path) == [sample]
    calibration = {"sample": data, "labels": {"faithfulness": True}}
    path = tmp_path / f"calibration.{suffix}"
    path.write_text(
        json.dumps([calibration]) if suffix == "json" else json.dumps(calibration) + "\n"
    )
    loaded = load_calibration_dataset(path)
    assert loaded[0].sample == sample
    assert loaded[0].labels == {"faithfulness": 1}


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("invalid_field", ["contexts", "trace"])
def test_dataset_validation_tracebacks_hide_row_values(
    tmp_path, sample, suffix, calibration, invalid_field
):
    sentinel = "PRIVATE_TEST_MARKER__dataset_validation"
    valid = sample.model_dump(mode="json")
    invalid = dict(valid)
    if invalid_field == "contexts":
        invalid["contexts"] = sentinel
        location = "contexts"
    else:
        invalid["trace"] = [{"name": "tool", "status": sentinel}]
        location = "trace.0.status"
    if calibration:
        valid = {"sample": valid, "labels": {"faithfulness": 1}}
        invalid = {"sample": invalid, "labels": {"faithfulness": 1}}
        location = f"sample.{location}"
    path = tmp_path / f"private.{suffix}"
    path.write_text(
        json.dumps([valid, invalid])
        if suffix == "json"
        else "\n" + json.dumps(valid) + "\n" + json.dumps(invalid) + "\n"
    )
    loader = load_calibration_dataset if calibration else load_dataset
    try:
        loader(path)
    except ValueError as exc:
        message = str(exc)
        formatted = traceback.format_exc()
    else:
        pytest.fail("Invalid row was accepted")

    row_index = 2 if suffix == "json" else 3
    kind = "calibration" if calibration else "evaluation"
    assert f"Invalid {kind} row {row_index}" in message
    assert location in message
    assert sentinel not in message
    assert sentinel not in formatted
    assert "ValidationError" not in formatted


@pytest.mark.parametrize(
    "suffix,payload,message",
    [
        ("json", "{}", "JSON datasets must contain an array of rows"),
        (
            "jsonl",
            '\n{"input": "Q", "response": "A"}\n{broken}',
            "Invalid JSON in data.jsonl at line 3",
        ),
        (
            "json",
            '[{"input": "Q", "response": "A", "typo": 1}]',
            "Invalid evaluation row 1, fields: ['typo']",
        ),
        ("txt", "anything", "Dataset must have a .json or .jsonl extension"),
    ],
)
def test_bad_datasets_have_actionable_errors(tmp_path, suffix, payload, message):
    path = tmp_path / f"data.{suffix}"
    path.write_text(payload)
    with pytest.raises(ValueError) as caught:
        load_dataset(path)
    assert str(caught.value) == message


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


@pytest.mark.parametrize("async_function", [False, True])
@pytest.mark.parametrize(
    "output_kind", ["generator", "async_generator", "coroutine", "future", "awaitable"]
)
async def test_decorator_rejects_unmaterialized_outputs_before_mapping(
    backend, async_function, output_kind
):
    executed = []
    builder_calls = []

    def stream():
        executed.append("generator")
        yield "piece"

    async def async_stream():
        executed.append("async_generator")
        yield "piece"

    async def pending():
        executed.append("coroutine")
        return "piece"

    class AwaitableOutput:
        def __await__(self):
            executed.append("awaitable")
            yield
            return "piece"

    if output_kind == "generator":
        output = stream()
    elif output_kind == "async_generator":
        output = async_stream()
    elif output_kind == "coroutine":
        output = pending()
    elif output_kind == "future":
        output = asyncio.get_running_loop().create_future()
    else:
        output = AwaitableOutput()

    def agent():
        return output

    async def async_agent():
        return output

    def build_samples(*args):
        builder_calls.append(args)
        return EvaluationSample(input="Q", response="A")

    wrapped = evaluated_by(Evaluator(backend=backend), sample_builder=build_samples)(
        async_agent if async_function else agent
    )
    try:
        with pytest.raises(TypeError, match="materialized before evaluation"):
            if async_function:
                await wrapped()
            else:
                wrapped()
        assert not executed
        assert not builder_calls
        assert not backend.calls
        assert backend.sessions == 0
        # Rejection leaves these caller-owned objects available for explicit handling.
        if inspect.isgenerator(output):
            assert inspect.getgeneratorstate(output) == inspect.GEN_CREATED
        elif inspect.isasyncgen(output):
            assert output.ag_frame is not None
        elif inspect.iscoroutine(output):
            assert inspect.getcoroutinestate(output) == inspect.CORO_CREATED
        elif isinstance(output, asyncio.Future):
            assert not output.done()
    finally:
        if inspect.isgenerator(output) or inspect.iscoroutine(output):
            output.close()
        elif inspect.isasyncgen(output):
            await output.aclose()
        elif isinstance(output, asyncio.Future):
            output.cancel()


@pytest.mark.parametrize("async_function", [False, True])
async def test_decorator_accepts_explicitly_materialized_streams(backend, async_function):
    native = list(piece for piece in ("Done",))

    def agent():
        return native

    async def async_agent():
        return native

    wrapped = evaluated_by(
        Evaluator(backend=backend),
        sample_builder=lambda output, args, kwargs: EvaluationSample(
            input="Run a task", response="".join(output)
        ),
    )(async_agent if async_function else agent)
    result = await wrapped() if async_function else wrapped()
    assert result.output is native
    assert isinstance(result.evaluation, SampleResult)
    assert result.evaluation.passed


def test_saved_isotonic_calibration_loads_and_thresholds_without_sklearn(
    monkeypatch, scoring_backend, tmp_path
):
    config = CalibrationConfig(
        enabled=True, algorithm="isotonic", min_samples=20, min_validation_samples=10
    )
    fitted = EvaluationPipeline(backend=scoring_backend, calibration=config)
    fitted.fit(calibrated_rows(), validation_data=calibrated_rows("holdout"))
    artifact = tmp_path / "calibration.json"
    fitted.save_calibration(artifact)
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    restored = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.7)], backend=scoring_backend, calibration=config
    ).load_calibration(artifact)
    result = restored.evaluate_one(EvaluationSample(input="production query", response="0.8"))
    assert result.metrics["answer_relevancy"].raw_score == 0.8
    assert result.metrics["answer_relevancy"].calibrated_probability == pytest.approx(0.6)
    assert result.passed is False
    calls_before = len(scoring_backend.calls)
    with pytest.raises(CalibrationError, match=r"typed_evals\[calibration\]"):
        IsotonicCalibrator.fit([0.1, 0.2, 0.7, 0.8], [0, 0, 1, 1])
    assert len(scoring_backend.calls) == calls_before


def test_cli_invalid_dataset_returns_error_without_network(tmp_path, monkeypatch, backend, capsys):
    import typed_evals.cli as cli

    monkeypatch.setattr(cli, "JevBackend", lambda **kwargs: backend)
    assert (
        main(["evaluate", str(tmp_path / "missing.json"), "--output", str(tmp_path / "x.json")])
        == 2
    )
    assert not backend.calls
    assert capsys.readouterr().err.startswith("typed_evals failed (FileNotFoundError).")
