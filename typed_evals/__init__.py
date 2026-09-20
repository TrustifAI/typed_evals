"""Jev-powered evaluation, with optional per-metric isotonic calibration."""

from typed_evals.backends import Backend, JevBackend, JudgeResponse, JudgeSession
from typed_evals.calibration import CalibrationBundle, CalibrationConfig, IsotonicCalibrator
from typed_evals.data.datasets import load_calibration_dataset, load_dataset
from typed_evals.data.models import (
    CalibrationExample,
    EvaluationReport,
    EvaluationSample,
    MetricResult,
    SampleResult,
    ToolCall,
    ToolProposal,
)
from typed_evals.evaluation.decorators import EvaluatedResponse, evaluated_by
from typed_evals.evaluation.evaluator import Evaluator, aevaluate, evaluate
from typed_evals.evaluation.pipeline import EvaluationPipeline
from typed_evals.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextRelevance,
    Faithfulness,
    Metric,
    PolicyCompliance,
    TaskCompletion,
    ToolAccuracy,
    ToolGrounding,
    ToolSafety,
    list_metrics,
)
from typed_evals.runtime.agents import GuardedAgent, guarded_agent
from typed_evals.runtime.guards import (
    GuardDecision,
    GuardedResponse,
    GuardPolicy,
    GuardrailViolation,
    RuntimeGuard,
    guarded_by,
)
from typed_evals.runtime.tools import guard_tool

__version__ = "0.1.0"
__all__ = [
    "AnswerCorrectness",
    "AnswerRelevancy",
    "Backend",
    "CalibrationBundle",
    "CalibrationConfig",
    "CalibrationExample",
    "ContextRelevance",
    "EvaluatedResponse",
    "EvaluationPipeline",
    "EvaluationReport",
    "EvaluationSample",
    "Evaluator",
    "Faithfulness",
    "GuardDecision",
    "GuardedAgent",
    "GuardedResponse",
    "GuardPolicy",
    "GuardrailViolation",
    "IsotonicCalibrator",
    "JevBackend",
    "JudgeResponse",
    "JudgeSession",
    "Metric",
    "MetricResult",
    "PolicyCompliance",
    "RuntimeGuard",
    "SampleResult",
    "TaskCompletion",
    "ToolAccuracy",
    "ToolCall",
    "ToolGrounding",
    "ToolProposal",
    "ToolSafety",
    "aevaluate",
    "evaluate",
    "evaluated_by",
    "guarded_by",
    "guarded_agent",
    "guard_tool",
    "list_metrics",
    "load_calibration_dataset",
    "load_dataset",
]
