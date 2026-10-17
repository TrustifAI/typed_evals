"""Typed judge evaluation, with optional per-metric probability calibration."""

from importlib.metadata import version as _distribution_version

from typed_evals.backends import (
    Backend,
    BaseBackend,
    JevBackend,
    JudgeResponse,
    JudgeSession,
    OpenAIDecisionsBackend,
    SystemOneBackend,
)
from typed_evals.calibration import (
    CalibrationBundle,
    CalibrationConfig,
    IsotonicCalibrator,
    ThresholdDiagnostics,
    VennAbersCalibrator,
)
from typed_evals.data.datasets import load_calibration_dataset, load_dataset
from typed_evals.data.models import (
    CalibrationExample,
    ContextRelevanceDetails,
    ContextUnitResult,
    EvaluationReport,
    EvaluationSample,
    ImageInput,
    MetricResult,
    SampleResult,
    ToolCall,
    ToolProposal,
)
from typed_evals.evaluation.decorators import EvaluatedResponse, evaluated_by
from typed_evals.evaluation.evaluator import Evaluator, aevaluate, evaluate
from typed_evals.evaluation.pipeline import EvaluationPipeline
from typed_evals.evaluation.validation import (
    MetricValidationReport,
    ValidationReport,
    avalidate_metrics,
    validate_metrics,
)
from typed_evals.metrics import (
    AnswerCompleteness,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextRelevance,
    ContextSufficiency,
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

__version__ = _distribution_version("typed_evals")
__all__ = [
    "AnswerCorrectness",
    "AnswerCompleteness",
    "AnswerRelevancy",
    "Backend",
    "BaseBackend",
    "CalibrationBundle",
    "CalibrationConfig",
    "CalibrationExample",
    "ContextRelevance",
    "ContextRelevanceDetails",
    "ContextSufficiency",
    "ContextUnitResult",
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
    "ImageInput",
    "IsotonicCalibrator",
    "JevBackend",
    "JudgeResponse",
    "JudgeSession",
    "Metric",
    "MetricResult",
    "MetricValidationReport",
    "OpenAIDecisionsBackend",
    "PolicyCompliance",
    "RuntimeGuard",
    "SampleResult",
    "SystemOneBackend",
    "TaskCompletion",
    "ToolAccuracy",
    "ToolCall",
    "ToolGrounding",
    "ToolProposal",
    "ToolSafety",
    "ThresholdDiagnostics",
    "VennAbersCalibrator",
    "ValidationReport",
    "avalidate_metrics",
    "aevaluate",
    "evaluate",
    "evaluated_by",
    "guarded_by",
    "guarded_agent",
    "guard_tool",
    "list_metrics",
    "load_calibration_dataset",
    "load_dataset",
    "validate_metrics",
]
