import asyncio
import json
import warnings

import pytest
from conftest import FakeBackend

from typed_evals import (
    AnswerCorrectness,
    AnswerRelevancy,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    Faithfulness,
    JudgeResponse,
    Metric,
    ToolAccuracy,
    ToolProposal,
)
from typed_evals.errors import InvalidAnswerError, MissingInputError


def test_all_metrics_share_one_call_and_labels_never_reach_jev(sample, backend):
    private = sample.model_copy(update={"metadata": {"label": "secret"}, "group_id": "group"})
    report = Evaluator([Faithfulness(), AnswerCorrectness()], backend=backend).evaluate([private])
    assert len(backend.calls) == 1
    state, questions = backend.calls[0]
    assert set(questions) == {"faithfulness", "answer_correctness"}
    assert set(state) == {"input", "response", "contexts", "reference"}
    assert "secret" not in str(state)
    assert report.results[0].passed
    assert report.summary["faithfulness"].mean_score == 0.8
    assert not report.calibrated


def test_missing_inputs_preflight_the_entire_batch(sample, backend):
    evaluator = Evaluator([Faithfulness()], backend=backend)
    bad = EvaluationSample(input="other", response="x")
    with pytest.raises(MissingInputError):
        evaluator.evaluate([sample, bad])
    assert not backend.calls
    assert backend.sessions == 0


def test_skips_do_not_become_zero_or_passed(backend):
    report = Evaluator([Faithfulness()], backend=backend, missing="skip").evaluate(
        [EvaluationSample(input="Q", response="A")]
    )
    assert report.results[0].metrics["faithfulness"].status == "skipped"
    assert report.results[0].passed is None
    assert report.summary["faithfulness"].mean_score is None
    assert report.summary["faithfulness"].skipped == 1
    assert backend.sessions == 0


def test_missing_answer_can_be_recorded_without_hiding_successful_metrics(sample):
    backend = FakeBackend(lambda state, qs: {"faithfulness": {"type": "noul", "noul": 0.7}})
    report = Evaluator(
        [Faithfulness(), AnswerRelevancy()], backend=backend, errors="record"
    ).evaluate([sample])
    assert report.results[0].metrics["faithfulness"].score == 0.7
    assert report.results[0].metrics["answer_relevancy"].status == "error"
    assert report.results[0].passed is None
    assert report.summary["answer_relevancy"].errors == 1


def test_missing_answer_raises_by_default(sample):
    backend = FakeBackend(lambda state, qs: {})
    with pytest.raises(InvalidAnswerError):
        Evaluator(backend=backend).evaluate([sample])


def test_provider_error_record_does_not_leak_request_content(sample):
    def fail(state, qs):
        raise RuntimeError("api-key-secret and private response")

    report = Evaluator(backend=FakeBackend(fail), errors="record").evaluate([sample])
    assert "secret" not in str(report.to_dict())
    assert report.results[0].metrics["answer_relevancy"].status == "error"


async def test_concurrency_is_bounded_and_out_of_order_answers_stay_with_their_samples():
    started = [asyncio.Event() for _ in range(6)]
    release = [asyncio.Event() for _ in range(6)]
    finished = [asyncio.Event() for _ in range(6)]
    completion_order = []

    class ControlledBackend(FakeBackend):
        async def judge(self, state, questions):
            index = int(state["input"])
            self.calls.append((state, questions))
            self.active += 1
            self.peak = max(self.peak, self.active)
            started[index].set()
            try:
                await release[index].wait()
                completion_order.append(index)
                return JudgeResponse(
                    model=self.model,
                    answers={"answer_relevancy": {"type": "noul", "noul": index / 10}},
                )
            finally:
                self.active -= 1
                finished[index].set()

    backend = ControlledBackend()
    samples = [EvaluationSample(input=str(i), response="A", id=f"row-{i}") for i in range(6)]
    task = asyncio.create_task(Evaluator(backend=backend, max_concurrency=3).aevaluate(samples))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started[:3])), 5)
        assert len(backend.calls) == backend.active == 3
        assert not any(event.is_set() for event in started[3:])
        for index in (2, 1, 0, 5, 4, 3):
            await asyncio.wait_for(started[index].wait(), 5)
            release[index].set()
            await asyncio.wait_for(finished[index].wait(), 5)
        report = await asyncio.wait_for(task, 5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert completion_order == [2, 1, 0, 5, 4, 3]
    assert backend.peak == 3
    assert [(row.sample_id, row.metrics["answer_relevancy"].score) for row in report.results] == [
        (f"row-{i}", i / 10) for i in range(6)
    ]
    assert backend.active == 0
    assert backend.sessions == backend.closed == 1


@pytest.mark.parametrize("evaluator_type", [Evaluator, EvaluationPipeline])
async def test_tool_results_identify_each_invocation_in_reports(evaluator_type, tmp_path):
    invocations = [("call_001", "lookup"), ("call_002", "search"), ("call_003", "lookup")]
    samples = [
        EvaluationSample(
            id=call_id,
            input="Find item 42",
            response="",
            contexts=("lookup(item_id: int) and search(item_id: int) find items.",),
            proposed_tool_call=ToolProposal(name=tool_name, arguments={"item_id": 42}),
        )
        for call_id, tool_name in invocations
    ]
    evaluator = evaluator_type([ToolAccuracy()], backend=FakeBackend(delay=0.001))
    report = await evaluator.aevaluate(samples)
    assert [(row.sample_id, row.tool_name) for row in report.results] == invocations
    assert all(row.metrics["tool_accuracy"].name == "tool_accuracy" for row in report.results)

    path = tmp_path / "tool-report.json"
    report.save(path)
    data = json.loads(path.read_text())
    assert [(row["sample_id"], row["tool_name"]) for row in data["results"]] == invocations


@pytest.mark.parametrize("status", ["skipped", "error"])
def test_tool_identity_is_preserved_when_metric_is_unavailable(status):
    sample = EvaluationSample(
        id="call_001",
        input="Find item 42",
        response="",
        contexts=None if status == "skipped" else ("lookup(item_id: int) finds an item.",),
        proposed_tool_call=ToolProposal(name="lookup", arguments={"item_id": 42}),
    )
    evaluator = Evaluator(
        [ToolAccuracy()],
        backend=FakeBackend(lambda state, questions: {}),
        missing="skip",
        errors="record",
    )
    result = evaluator.evaluate_one(sample)
    assert result.metrics["tool_accuracy"].status == status
    assert result.model_dump(mode="json")["tool_name"] == "lookup"
    assert result.sample_id == "call_001"


@pytest.mark.parametrize("sync", [False, True])
async def test_failure_cancels_other_workers_and_closes_session(sync):
    class Failing(FakeBackend):
        def __init__(self):
            super().__init__()
            self.started = None
            self.cancelled = False

        async def judge(self, state, questions):
            # The sync entry point can run in another thread's loop. Create this
            # synchronization primitive in the loop that actually judges the batch.
            if self.started is None:
                self.started = asyncio.Event()
            if state["input"] == "fail":
                await asyncio.wait_for(self.started.wait(), 5)
                raise RuntimeError("failed")
            self.active += 1
            self.started.set()
            try:
                await asyncio.wait_for(asyncio.Event().wait(), 5)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                self.active -= 1

    backend = Failing()
    evaluator = Evaluator(backend=backend)
    samples = [
        EvaluationSample(input="wait", response="A"),
        EvaluationSample(input="fail", response="A"),
    ]
    with pytest.raises(RuntimeError, match="failed"):
        if sync:
            evaluator.evaluate(samples)
        else:
            await evaluator.aevaluate(samples)
    assert backend.active == 0
    assert backend.cancelled
    assert backend.closed == 1


@pytest.mark.parametrize("evaluator_type", [Evaluator, EvaluationPipeline])
async def test_sync_api_works_in_running_loop(sample, backend, evaluator_type):
    loop = asyncio.get_running_loop()
    evaluator = evaluator_type(backend=backend)

    # Repeated sync calls and an async call can share the same evaluator in a notebook.
    result = evaluator.evaluate_one(sample)
    report = evaluator.evaluate([sample])
    async_result = await evaluator.aevaluate_one(sample)

    assert result.sample_id == report.results[0].sample_id == async_result.sample_id == sample.id
    assert result.metrics == report.results[0].metrics == async_result.metrics
    assert result.passed
    assert backend.sessions == backend.closed == 3
    await asyncio.sleep(0)
    assert asyncio.get_running_loop() is loop


def test_empty_batch_does_not_open_api_session(backend):
    report = Evaluator(backend=backend).evaluate([])
    assert report.results == ()
    assert backend.sessions == 0


@pytest.mark.parametrize("evaluator_type", [Evaluator, EvaluationPipeline])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unused_contexts_warn_once_per_batch(evaluator_type, asynchronous, backend):
    samples = [
        EvaluationSample(input="Q", response="A", contexts=("private context",)),
        EvaluationSample(input="Q2", response="A2"),
        EvaluationSample(input="Q3", response="A3", contexts=("other context",)),
    ]
    evaluator = evaluator_type([AnswerRelevancy()], backend=backend)
    with pytest.warns(UserWarning, match="contexts were supplied for 2 sample") as caught:
        report = await evaluator.aevaluate(samples) if asynchronous else evaluator.evaluate(samples)
    assert len(caught) == 1
    assert "preset='rag'" in str(caught[0].message)
    assert "private context" not in str(caught[0].message)
    assert len(report.results) == len(samples)
    assert all("contexts" not in state for state, _ in backend.calls)


@pytest.mark.parametrize("selection", ["default", "rag", "custom"])
def test_used_or_absent_contexts_do_not_warn(selection, sample, backend):
    if selection == "default":
        samples = [sample.model_copy(update={"contexts": ()})]
        evaluator = Evaluator(backend=backend)
    elif selection == "rag":
        samples = [sample]
        evaluator = Evaluator(preset="rag", backend=backend)
    else:
        samples = [sample]
        metric = Metric(
            name="custom_context_check",
            instructions="Is the answer grounded in contexts?",
            required_fields=("input", "response", "contexts"),
            pass_definition="The answer is grounded in contexts",
        )
        evaluator = Evaluator([metric], backend=backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluator.evaluate(samples)
    if selection != "default":
        assert backend.calls[0][0]["contexts"] == list(sample.contexts)


@pytest.mark.parametrize("include_relevancy", [False, True])
def test_skipped_context_metric_does_not_consume_evidence(include_relevancy, sample, backend):
    context_metric = Metric(
        name="reference_grounding",
        instructions="Is the answer grounded in contexts and consistent with the reference?",
        required_fields=("input", "response", "contexts", "reference"),
        pass_definition="The answer meets both checks",
    )
    metrics = [context_metric, AnswerRelevancy()] if include_relevancy else [context_metric]
    without_reference = sample.model_copy(update={"reference": None})
    evaluator = Evaluator(metrics, backend=backend, missing="skip")
    with pytest.warns(UserWarning, match="contexts were supplied for 1 sample") as caught:
        report = evaluator.evaluate([sample, without_reference])
    assert len(caught) == 1
    assert report.results[1].metrics[context_metric.name].status == "skipped"
    assert backend.calls[0][0]["contexts"] == list(sample.contexts)
    assert len(backend.calls) == (2 if include_relevancy else 1)
    if include_relevancy:
        assert "contexts" not in backend.calls[1][0]


def test_unused_context_warning_is_emitted_before_opening_a_session(sample, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="contexts were supplied"):
            Evaluator(backend=backend).evaluate([sample])
    assert backend.sessions == 0


def test_saved_report_excludes_unavailable_metrics_from_means_and_pass_rates(tmp_path):
    def answer(state, questions):
        if state["response"] == "error":
            raise RuntimeError("private provider payload")
        return {"faithfulness": {"type": "noul", "noul": float(state["response"])}}

    backend = FakeBackend(answer)
    samples = [
        EvaluationSample(input="Q", response=value, contexts=("Evidence",), id=value)
        for value in ("0.8", "0.2", "error")
    ] + [EvaluationSample(input="Q", response="missing", id="skipped")]
    report = Evaluator(
        [Faithfulness(threshold=0.8)], backend=backend, missing="skip", errors="record"
    ).evaluate(samples)
    path = tmp_path / "nested" / "report.json"
    report.save(path)
    data = json.loads(path.read_text())

    assert data["summary"]["faithfulness"] == {
        "evaluated": 2,
        "skipped": 1,
        "errors": 1,
        "passed": 1,
        "mean_raw_score": 0.5,
        "mean_score": 0.5,
        "pass_rate": 0.5,
    }
    assert [row["sample_id"] for row in data["results"]] == ["0.8", "0.2", "error", "skipped"]
    assert [row["passed"] for row in data["results"]] == [True, False, None, None]
    for row, raw in zip(data["results"], (0.8, 0.2, None, None), strict=True):
        metric = row["metrics"]["faithfulness"]
        assert metric["raw_score"] == metric["score"] == raw
        assert metric["calibrated_probability"] is None
        assert row["tool_name"] is None
    assert "private provider payload" not in path.read_text()
    assert len(backend.calls) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metrics": []},
        {"metrics": [Faithfulness(), Faithfulness()]},
        {"max_concurrency": 0},
        {"max_concurrency": True},
        {"missing": "ignore"},
        {"errors": "ignore"},
    ],
)
def test_invalid_evaluator_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Evaluator(**kwargs)
