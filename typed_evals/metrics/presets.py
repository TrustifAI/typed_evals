"""Explicit metric panels; evidence never silently changes the selected checks."""

from typing import Literal

from typed_evals.metrics import (
    AnswerRelevancy,
    ContextRelevance,
    Faithfulness,
    Metric,
    TaskCompletion,
    ToolGrounding,
)

Preset = Literal["response", "rag", "agent"]


def preset_metrics(preset: Preset) -> tuple[Metric, ...]:
    panels = {
        "response": (AnswerRelevancy,),
        "rag": (Faithfulness, AnswerRelevancy, ContextRelevance),
        "agent": (TaskCompletion, ToolGrounding),
    }
    if not isinstance(preset, str) or preset not in panels:
        raise ValueError("preset must be 'response', 'rag', or 'agent'")
    return tuple(factory() for factory in panels[preset])
