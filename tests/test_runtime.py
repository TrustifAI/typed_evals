import asyncio
import json
import threading

import pytest
from conftest import FakeBackend, binary_answer, calibrated_rows
from pydantic import ValidationError

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    EvaluationPipeline,
    EvaluationSample,
    Evaluator,
    Faithfulness,
    GuardPolicy,
    GuardrailViolation,
    PolicyCompliance,
    RuntimeGuard,
    TaskCompletion,
    ToolAccuracy,
    ToolProposal,
    ToolSafety,
    guarded_by,
)
from typed_evals._utils import digest
from typed_evals.data.models import MetricResult, SampleResult
from typed_evals.errors import CalibrationError


def judge(score=0.8, metrics=None, **kwargs):
    return Evaluator(
        metrics,
        backend=FakeBackend(
            lambda state, questions: {
                name: binary_answer(question, score) for name, question in questions.items()
            }
        ),
        **kwargs,
    )


def tool_sample(**overrides):
    return EvaluationSample(
        **{
            "input": "Read ticket T-42",
            "response": "",
            "contexts": ["read_ticket(ticket_id: str) returns a ticket. The user can read T-42."],
            "proposed_tool_call": ToolProposal(name="read_ticket", arguments={"ticket_id": "T-42"}),
            **overrides,
        }
    )


@pytest.mark.parametrize("action", ["block", "retry", "escalate"])
def test_denied_actions_never_invoke_operation(action, sample):
    calls = []
    guard = RuntimeGuard({"step": GuardPolicy(judge(0.1), on_fail=action)})
    with pytest.raises(GuardrailViolation) as caught:
        guard.run("step", sample, lambda: calls.append("executed"))
    assert not calls
    assert caught.value.decision.action == action
    assert caught.value.decision.failed_metrics == ("answer_relevancy",)
    assert not caught.value.decision.allowed
    assert caught.value.decision.to_dict()["evaluation"]["passed"] is False


def test_check_only_returns_decision_and_enforce_raises(sample):
    guard = RuntimeGuard({"step": GuardPolicy(judge(0.1))})
    decision = guard.check("step", sample)
    assert not decision.allowed
    with pytest.raises(GuardrailViolation):
        decision.require_allowed()
    with pytest.raises(GuardrailViolation):
        guard.enforce("step", sample)


@pytest.mark.parametrize("score,action,suspected", [(0.1, "annotate", True), (0.9, "allow", False)])
def test_response_annotation_preserves_native_output_and_metadata(sample, score, action, suspected):
    native = {"answer": sample.response, "metadata": {"caller": "preserved"}}
    guard = RuntimeGuard(
        {"response": GuardPolicy(judge(score, [Faithfulness()]), on_fail="annotate")}
    )
    result = guard.respond("response", sample, native)
    assert result.output is native
    assert native["metadata"] == {"caller": "preserved"}
    assert result.decisions[0].action == action
    assert result.metadata["jev"]["hallucination_suspected"] is suspected
    event = result.metadata["jev"]["decisions"][0]
    assert event["evaluation"]["metrics"]["faithfulness"]["score"] == score
    assert event["evaluation"]["sample_hash"] == sample.content_hash
    assert json.loads(json.dumps(result.metadata)) == result.metadata


def test_missing_grounding_is_unknown_not_a_clean_bill_of_health():
    evaluator = judge(metrics=[Faithfulness()], missing="skip")
    guard = RuntimeGuard({"response": GuardPolicy(evaluator, on_error="annotate")})
    sample = EvaluationSample(input="Q", response="Unsupported")
    result = guard.respond("response", sample, "Unsupported")
    assert result.metadata["jev"]["hallucination_suspected"] is None
    assert result.decisions[0].unavailable_metrics == ("faithfulness",)


@pytest.mark.parametrize("missing", ["raise", "skip"])
def test_missing_evidence_blocks_by_default(missing):
    guard = RuntimeGuard({"step": GuardPolicy(judge(metrics=[Faithfulness()], missing=missing))})
    sample = EvaluationSample(input="Q", response="A")
    with pytest.raises(GuardrailViolation):
        guard.run("step", sample, lambda: pytest.fail("must not execute"))


@pytest.mark.parametrize("errors", ["raise", "record"])
def test_judge_failure_blocks_without_leaking_exception_content(errors, sample):
    def fail(state, questions):
        raise RuntimeError("secret-api-key and private request")

    guard = RuntimeGuard({"step": GuardPolicy(Evaluator(backend=FakeBackend(fail), errors=errors))})
    with pytest.raises(GuardrailViolation) as caught:
        guard.run("step", sample, lambda: pytest.fail("must not execute"))
    assert "secret" not in json.dumps(caught.value.decision.to_dict())
    assert "RuntimeError" in json.dumps(caught.value.decision.to_dict())


@pytest.mark.parametrize("answers", [{}, {"answer_relevancy": {"type": "noul", "noul": 10}}])
def test_incomplete_or_malformed_judgment_cannot_approve(answers, sample):
    evaluator = Evaluator(backend=FakeBackend(lambda state, qs: answers), errors="record")
    guard = RuntimeGuard({"step": GuardPolicy(evaluator)})
    assert guard.check("step", sample).action == "block"


def test_empty_evaluator_results_cannot_approve(sample):
    class EmptyEvaluator:
        async def aevaluate_one(self, sample):
            return SampleResult(
                sample_id="0", sample_hash="x", model=None, metrics={}, elapsed_ms=0
            )

    guard = RuntimeGuard({"step": GuardPolicy(EmptyEvaluator())})
    decision = guard.check("step", sample)
    assert decision.action == "block"
    assert "InvalidAnswerError" in decision.error


def test_missing_grounding_score_stays_unknown_even_if_custom_evaluator_claims_pass(sample):
    class IncompleteEvaluator:
        async def aevaluate_one(self, sample):
            return SampleResult(
                sample_id="0",
                sample_hash=sample.content_hash,
                model=None,
                elapsed_ms=0,
                metrics={
                    "faithfulness": MetricResult(
                        name="faithfulness", status="ok", threshold=0.5, passed=True
                    )
                },
            )

    guard = RuntimeGuard({"response": GuardPolicy(IncompleteEvaluator(), on_error="annotate")})
    result = guard.respond("response", sample, sample.response)
    assert result.decisions[0].unavailable_metrics == ("faithfulness",)
    assert result.metadata["jev"]["hallucination_suspected"] is None


def test_annotated_judge_error_keeps_aggregate_grounding_unknown(sample):
    def fail(state, questions):
        raise RuntimeError("unavailable")

    guard = RuntimeGuard(
        {
            "before": GuardPolicy(Evaluator(backend=FakeBackend(fail)), on_error="annotate"),
            "after": GuardPolicy(judge(0.9, [Faithfulness()])),
        }
    )
    result = guard.run(
        "before", sample, lambda: "A", after="after", sample_builder=lambda output, state: state
    )
    assert result.decisions[-1].hallucination_suspected is False
    assert result.metadata["jev"]["hallucination_suspected"] is None


def test_per_metric_policies_combine_with_most_restrictive_action(sample):
    evaluator = judge(0.1, [Faithfulness(), AnswerRelevancy()])
    policy = GuardPolicy(
        evaluator, on_fail="annotate", metric_actions={"answer_relevancy": "retry"}
    )
    decision = RuntimeGuard({"response": policy}).check("response", sample)
    assert decision.action == "retry"
    assert decision.hallucination_suspected is True
    assert set(decision.failed_metrics) == {"faithfulness", "answer_relevancy"}
    unknown = GuardPolicy(evaluator, metric_actions={"typo": "annotate"})
    with pytest.raises(ValueError, match="unknown metrics"):
        RuntimeGuard({"response": unknown}).check("response", sample)


def test_unavailable_metric_keeps_default_block_despite_annotation_policy():
    evaluator = judge(0.1, [Faithfulness(), AnswerRelevancy()], missing="skip")
    guard = RuntimeGuard({"step": GuardPolicy(evaluator, on_fail="annotate")})
    decision = guard.check("step", EvaluationSample(input="Q", response="A"))
    assert decision.action == "block"
    assert decision.failed_metrics == ("answer_relevancy",)
    assert decision.unavailable_metrics == ("faithfulness",)


async def test_timeout_cleans_up_and_never_starts_tool(sample):
    backend = FakeBackend(delay=10)
    guard = RuntimeGuard({"step": GuardPolicy(Evaluator(backend=backend), timeout=0.01)})
    with pytest.raises(GuardrailViolation) as caught:
        await guard.arun("step", sample, lambda: pytest.fail("must not execute"))
    assert caught.value.decision.error == "Evaluation failed: TimeoutError"
    assert backend.active == 0
    assert backend.sessions == backend.closed == 1


async def test_cancellation_propagates_before_tool_starts(sample):
    entered = asyncio.Event()

    class WaitingBackend(FakeBackend):
        async def judge(self, state, questions):
            entered.set()
            return await super().judge(state, questions)

    backend = WaitingBackend(delay=10)
    guard = RuntimeGuard({"step": GuardPolicy(Evaluator(backend=backend), on_error="annotate")})
    task = asyncio.create_task(guard.arun("step", sample, lambda: pytest.fail("must not execute")))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.sessions == backend.closed == 1


def test_tool_dispatch_uses_reviewed_proposal_once_and_postcheck_observes_result():
    calls = []
    before = judge(0.95, [ToolSafety(policy="Only read authorized tickets."), ToolAccuracy()])
    after = judge(0.95, [TaskCompletion()])
    guard = RuntimeGuard({"before_tool": GuardPolicy(before), "after_tool": GuardPolicy(after)})
    native = {"ticket_id": "T-42", "title": "Export failure"}

    def read_ticket(ticket_id):
        calls.append(ticket_id)
        return native

    result = guard.call_tool(
        {"read_ticket": read_ticket},
        tool_sample(expected_outcome="Ticket T-42 was retrieved."),
        after="after_tool",
    )
    assert calls == ["T-42"]
    assert result.output is native
    assert len(result.decisions) == 2
    decisions = result.metadata["jev"]["decisions"]
    assert decisions[0]["evaluation"]["tool_name"] == "read_ticket"
    assert decisions[1]["evaluation"]["tool_name"] is None
    before_state, questions = before.backend.calls[0]
    assert before_state["proposed_tool_call"]["arguments"] == {"ticket_id": "T-42"}
    assert "trace" not in before_state
    assert set(questions) == {"tool_safety", "tool_accuracy"}
    trace = after.backend.calls[0][0]["trace"]
    assert trace == [
        {
            "name": "read_ticket",
            "arguments": {"ticket_id": "T-42"},
            "output": native,
            "status": "success",
            "error": None,
        }
    ]


async def test_external_mutation_during_review_cannot_change_executed_arguments():
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingBackend(FakeBackend):
        async def judge(self, state, questions):
            entered.set()
            await release.wait()
            return await super().judge(state, questions)

    sample = tool_sample(
        proposed_tool_call=ToolProposal(name="save", arguments={"item": {"id": "safe"}})
    )
    registry = {"save": lambda item: item["id"]}
    backend = WaitingBackend()
    guard = RuntimeGuard({"before_tool": GuardPolicy(Evaluator([ToolAccuracy()], backend=backend))})
    task = asyncio.create_task(guard.acall_tool(registry, sample))
    await entered.wait()
    sample.proposed_tool_call.arguments["item"]["id"] = "unsafe"
    registry["save"] = lambda item: pytest.fail("registry changed after review started")
    release.set()
    result = await task
    assert result.output == "safe"
    assert backend.calls[0][0]["proposed_tool_call"]["arguments"]["item"]["id"] == "safe"


@pytest.mark.parametrize("async_tool", [False, True])
async def test_async_dispatch_supports_native_async_and_sync_tools(async_tool):
    event_loop_thread = threading.get_ident()

    async def async_read(ticket_id):
        assert threading.get_ident() == event_loop_thread
        return ticket_id

    def sync_read(ticket_id):
        assert threading.get_ident() != event_loop_thread
        return ticket_id

    guard = RuntimeGuard({"before_tool": GuardPolicy(judge())})
    result = await guard.acall_tool(
        {"read_ticket": async_read if async_tool else sync_read}, tool_sample()
    )
    assert result.output == "T-42"


def test_failed_tool_is_not_retried_or_postchecked():
    calls = []
    evaluator = judge()
    guard = RuntimeGuard(
        {"before_tool": GuardPolicy(evaluator), "after_tool": GuardPolicy(evaluator)}
    )

    def fail(ticket_id):
        calls.append(ticket_id)
        raise LookupError("tool failure")

    with pytest.raises(LookupError, match="tool failure"):
        guard.call_tool({"read_ticket": fail}, tool_sample(), after="after_tool")
    assert calls == ["T-42"]
    assert len(evaluator.backend.calls) == 1


def test_postcheck_denial_withholds_output_but_cannot_undo_side_effects(sample):
    calls = []
    guard = RuntimeGuard({"before": GuardPolicy(judge()), "after": GuardPolicy(judge(0.1))})
    with pytest.raises(GuardrailViolation) as caught:
        guard.run(
            "before",
            sample,
            lambda: calls.append("executed"),
            after="after",
            sample_builder=lambda out, state: state,
        )
    assert calls == ["executed"]
    assert caught.value.decision.checkpoint == "after"


def test_non_json_tool_outputs_can_be_mapped_for_postcheck():
    native = object()
    guard = RuntimeGuard(
        {
            "before_tool": GuardPolicy(judge()),
            "after_tool": GuardPolicy(judge(metrics=[TaskCompletion()])),
        }
    )
    result = guard.call_tool(
        {"read_ticket": lambda ticket_id: native},
        tool_sample(expected_outcome="Read ticket"),
        after="after_tool",
        output_mapper=lambda output: {"retrieved": output is native},
    )
    assert result.output is native


@pytest.mark.parametrize("case", ["name", "arguments", "proposal", "checkpoint", "after"])
def test_dispatch_configuration_errors_precede_execution_and_judgment(case):
    evaluator = judge()
    guard = RuntimeGuard({"before_tool": GuardPolicy(evaluator)})
    sample = tool_sample()
    options = {}
    if case == "name":
        sample = tool_sample(proposed_tool_call=ToolProposal(name="unregistered"))
    elif case == "arguments":
        sample = tool_sample(
            proposed_tool_call=ToolProposal(name="read_ticket", arguments={"typo": 1})
        )
    elif case == "proposal":
        sample = tool_sample(proposed_tool_call=None)
    elif case == "checkpoint":
        options["before"] = "typo"
    else:
        options["after"] = "typo"
    with pytest.raises((ValueError, TypeError)):
        guard.call_tool(
            {"read_ticket": lambda ticket_id: pytest.fail("must not execute")}, sample, **options
        )
    assert not evaluator.backend.calls


def test_sync_decorator_guards_both_boundaries_and_preserves_function_name(sample):
    guard = RuntimeGuard(
        {"before": GuardPolicy(judge()), "after": GuardPolicy(judge(0.1), on_fail="annotate")}
    )

    @guarded_by(
        guard,
        before="before",
        after="after",
        before_sample=lambda args, kwargs: sample,
        after_sample=lambda output, args, kwargs: sample,
    )
    def framework_call(question):
        return {"answer": question}

    result = framework_call("Q")
    assert result.output == {"answer": "Q"}
    assert [d.action for d in result.decisions] == ["allow", "annotate"]
    assert framework_call.__name__ == "framework_call"


async def test_async_after_only_decorator_and_response_helper(sample):
    guard = RuntimeGuard({"after": GuardPolicy(judge(0.1, [Faithfulness()]), on_fail="annotate")})

    @guarded_by(guard, after="after", after_sample=lambda output, args, kwargs: sample)
    async def framework_call():
        return "answer"

    result = await framework_call()
    assert result.output == "answer"
    assert result.metadata["jev"]["hallucination_suspected"] is True
    assert (await guard.arespond("after", sample, "answer")).metadata["jev"][
        "hallucination_suspected"
    ] is True


async def test_concurrent_checkpoints_do_not_share_decisions():
    guard = RuntimeGuard({"step": GuardPolicy(judge())})
    samples = [EvaluationSample(input=str(i), response="A", id=str(i)) for i in range(12)]
    results = await asyncio.gather(*[guard.arun("step", sample, lambda: "A") for sample in samples])
    assert [row.decisions[0].evaluation.sample_id for row in results] == [str(i) for i in range(12)]
    assert len({row.decisions[0].id for row in results}) == 12


async def test_sync_entrypoint_works_in_notebook_loop(sample):
    guard = RuntimeGuard({"step": GuardPolicy(judge())})
    result = guard.run("step", sample, lambda: "A")
    assert result.output == "A"
    assert (await guard.acheck("step", sample)).allowed


def test_calibrated_pipeline_scores_drive_runtime_policy(scoring_backend):
    pipeline = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.7)],
        backend=scoring_backend,
        calibration=CalibrationConfig(enabled=True, min_samples=20, min_validation_samples=10),
    )
    pipeline.fit(calibrated_rows(), validation_data=calibrated_rows("validation"))
    guard = RuntimeGuard({"step": GuardPolicy(pipeline)})
    decision = guard.check("step", EvaluationSample(input="independent", response="0.8"))
    metric = decision.evaluation.metrics["answer_relevancy"]
    assert metric.raw_score == 0.8
    assert metric.calibrated_probability == pytest.approx(0.6)
    assert decision.action == "block"
    assert decision.to_dict()["evaluation"]["metrics"]["answer_relevancy"][
        "score"
    ] == pytest.approx(0.6)


def test_calibration_configuration_error_is_fatal_even_when_errors_annotated(sample):
    pipeline = EvaluationPipeline(
        backend=FakeBackend(), calibration=CalibrationConfig(enabled=True)
    )
    guard = RuntimeGuard({"step": GuardPolicy(pipeline, on_error="annotate")})
    with pytest.raises(CalibrationError):
        guard.run("step", sample, lambda: pytest.fail("must not execute"))


@pytest.mark.parametrize(
    "options",
    [
        {"on_fail": "allow"},
        {"on_error": "ignore"},
        {"timeout": 0},
        {"timeout": True},
        {"timeout": float("nan")},
        {"metric_actions": {"test": "ignore"}},
    ],
)
def test_invalid_policy_options_fail_early(options):
    with pytest.raises(ValueError):
        GuardPolicy(judge(), **options)


def test_configuration_mappings_are_owned(sample):
    actions = {"answer_relevancy": "block"}
    policy = GuardPolicy(judge(0.1), metric_actions=actions)
    policies = {"step": policy}
    guard = RuntimeGuard(policies)
    actions["answer_relevancy"] = "annotate"
    policies.clear()
    assert guard.check("step", sample).action == "block"


def test_streaming_and_unawaited_functions_are_rejected_before_execution(sample):
    guard = RuntimeGuard({"step": GuardPolicy(judge())})

    def stream():
        yield "unsafe"

    async def async_stream():
        yield "unsafe"

    async def async_operation():
        pytest.fail("must not execute")

    for function in (stream, async_stream):
        with pytest.raises(TypeError, match="materialized"):
            guard.run("step", sample, function)
        with pytest.raises(TypeError, match="materialized"):
            guarded_by(guard, after="step", after_sample=lambda *args: sample)(function)
    with pytest.raises(TypeError, match="arun"):
        guard.run("step", sample, async_operation)


def test_proposals_cannot_masquerade_as_observed_results_and_legacy_hash_is_stable(sample):
    with pytest.raises(ValidationError):
        ToolProposal(name="delete", output="success")
    with pytest.raises(ValidationError):
        ToolProposal(name="  ")
    old_data = sample.model_dump(
        mode="json", exclude={"id", "group_id", "metadata", "proposed_tool_call"}
    )
    assert sample.content_hash == digest(old_data)
    proposed = EvaluationSample(
        **{**sample.model_dump(), "proposed_tool_call": ToolProposal(name="read")}
    )
    assert sample.content_hash != proposed.content_hash


@pytest.mark.parametrize("factory", [ToolSafety, PolicyCompliance])
def test_application_policy_is_required_and_part_of_calibration_identity(factory):
    with pytest.raises(ValueError):
        factory(policy=" ")
    original = factory(policy="Read only")
    assert original.fingerprint != factory(policy="Writes allowed").fingerprint
    assert original.fingerprint == factory(policy="Read only", threshold=0.1).fingerprint
    assert original.question().type == "noul"
