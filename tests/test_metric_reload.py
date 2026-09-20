"""Module reloads run in child interpreters so they cannot contaminate other tests."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import create_model

from typed_evals import Evaluator, Metric, ToolAccuracy


@pytest.mark.parametrize(
    "setup",
    [
        # Old Evaluator plus freshly created metrics, as after a notebook module reload.
        "importlib.reload(definitions)\nmetrics = make_metrics()",
        # Existing metric objects plus a newly imported evaluator/helper.
        "metrics = make_metrics()\nimportlib.reload(definitions)\nimportlib.reload(evaluator_module)",
        # Load the same source under an alias without depending on a src directory.
        "spec = importlib.util.spec_from_file_location(\n"
        "    'typed_evals.metrics._base_alias', definitions.__file__\n"
        ")\n"
        "alias = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = alias\n"
        "spec.loader.exec_module(alias)\n"
        "metrics = [alias.Metric.model_validate(m.model_dump()) for m in make_metrics()]",
    ],
)
def test_notebook_metric_class_generations_are_revalidated_and_can_gate_a_tool(setup):
    script = """
import importlib
import importlib.util
import sys
from contextlib import asynccontextmanager
from typed_evals import (
    EvaluationSample, Evaluator, GuardPolicy, JudgeResponse,
    RuntimeGuard, ToolAccuracy, ToolProposal, ToolSafety,
)
import typed_evals.metrics.base as definitions
import typed_evals.evaluation.evaluator as evaluator_module

def make_metrics():
    return [definitions.Metric.model_validate(metric.model_dump()) for metric in [
        ToolSafety(policy="Only read tickets belonging to the authenticated customer."),
        ToolAccuracy(threshold=0.85),
    ]]

SETUP

# Migrating a stale identity must still enforce the current schema.
if type(metrics[0]) is not definitions.Metric:
    from pydantic import ValidationError
    try:
        Evaluator([metrics[0].model_copy(update={"threshold": 2.0})])
    except ValidationError:
        pass
    else:
        raise AssertionError("Invalid migrated definition was accepted")

class Backend:
    model = "test-only"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        assert state["proposed_tool_call"]["arguments"] == {"ticket_id": "T-42"}
        return JudgeResponse(model=self.model, answers={
            name: {
                "type": "choice", "choice": "true", "confidence": 0.6,
                "probabilities": {"true": 0.99, "false": 0.01},
            } if question.type == "choice" else {"type": "noul", "noul": 0.99}
            for name, question in questions.items()
        })

evaluator = Evaluator(metrics, backend=Backend())
assert all(isinstance(metric, definitions.Metric) for metric in evaluator.metrics)
guard = RuntimeGuard({"before_tool": GuardPolicy(evaluator, timeout=10)})
result = guard.call_tool(
    {"read_ticket": lambda ticket_id: ticket_id},
    EvaluationSample(
        input="Read ticket T-42", response="",
        contexts=("read_ticket(ticket_id: str) reads a ticket; the customer owns T-42.",),
        proposed_tool_call=ToolProposal(name="read_ticket", arguments={"ticket_id": "T-42"}),
    ),
)
assert result.output == "T-42"
assert result.decisions[0].action == "allow"
assert evaluator.metrics[1].threshold == 0.85
metrics[0].criteria["true"] = "caller mutation"
assert evaluator.metrics[0].criteria["true"] != "caller mutation"
""".replace("SETUP", setup)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("invalid", [None, {}, ToolAccuracy, object()])
def test_invalid_metric_message_identifies_entry_without_echoing_value(invalid):
    with pytest.raises(ValueError, match=r"metrics\[1\] must be a typed_evals Metric instance"):
        Evaluator([ToolAccuracy(), invalid])


def test_foreign_model_called_metric_is_not_accepted():
    foreign = create_model(
        "Metric",
        **{name: (field.annotation, field.default) for name, field in Metric.model_fields.items()},
    )
    instance = foreign(**ToolAccuracy().model_dump())
    with pytest.raises(ValueError, match="typed_evals Metric instance"):
        Evaluator([instance])


def test_metric_subclass_behavior_is_preserved():
    class CustomMetric(Metric):
        def missing_fields(self, sample):
            return ["custom_evidence"]

    metric = CustomMetric(name="custom", instructions="Check", pass_definition="Pass")
    owned = Evaluator([metric]).metrics[0]
    assert type(owned) is CustomMetric
    assert owned is not metric
    assert owned.missing_fields(None) == ["custom_evidence"]
