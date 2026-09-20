from .agents import PolicyCompliance, TaskCompletion, ToolAccuracy, ToolGrounding, ToolSafety
from .rag import AnswerCorrectness, AnswerRelevancy, ContextRelevance, Faithfulness

# Metrics constructible with defaults, also used by the CLI.
BUILTIN_METRICS = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "answer_correctness": AnswerCorrectness,
    "context_relevance": ContextRelevance,
    "task_completion": TaskCompletion,
    "tool_grounding": ToolGrounding,
    "tool_accuracy": ToolAccuracy,
}

# These metrics require an application policy before they can be constructed.
_POLICY_METRICS = {
    "policy_compliance": PolicyCompliance,
    "tool_safety": ToolSafety,
}


def list_metrics() -> list[str]:
    """Return a new, alphabetically sorted list of all built-in metric names.

    Includes policy_compliance and tool_safety, whose constructors require a
    policy. Custom Metric instances are not registered here. No metrics are
    instantiated and no API credentials or network calls are needed.
    """
    return sorted(BUILTIN_METRICS.keys() | _POLICY_METRICS.keys())
