from .base import Metric


def Faithfulness(*, threshold: float = 0.5) -> Metric:
    return Metric(
        name="faithfulness",
        threshold=threshold,
        instructions="Is every material factual assertion in `response` supported by `contexts`?",
        criteria={
            "true": "Every factual assertion is supported by the supplied passages. Paraphrases "
            "are acceptable. A response with no factual assertions also satisfies this criterion.",
            "false": "At least one factual assertion contradicts the passages or cannot be supported "
            "by them. Plausibility and outside knowledge do not count as support.",
        },
        required_fields=("response", "contexts"),
        pass_definition="Every material factual assertion is supported by the retrieved passages.",
    )


def AnswerRelevancy(*, threshold: float = 0.5) -> Metric:
    return Metric(
        name="answer_relevancy",
        threshold=threshold,
        instructions="Does `response` directly address the information or action requested in `input`?",
        criteria={
            "true": "The response addresses the request, including a relevant clarification "
            "or explanation that the request cannot be completed.",
            "false": "The response is empty, off-topic, or avoids the request without a relevant explanation.",
        },
        pass_definition="The response directly addresses the user's request; factual correctness is separate.",
    )


def AnswerCorrectness(*, threshold: float = 0.5) -> Metric:
    return Metric(
        name="answer_correctness",
        threshold=threshold,
        instructions="Does `response` give the answer established by `reference` for `input`?",
        criteria={
            "true": "The response conveys the reference answer's essential facts with no "
            "contradiction. Equivalent wording and harmless extra detail are allowed.",
            "false": "The response contradicts or omits an essential part of the reference answer, "
            "adds a conflicting answer, or is empty.",
        },
        required_fields=("input", "response", "reference"),
        pass_definition="The response matches the essential answer in the supplied reference.",
    )


def ContextRelevance(*, threshold: float = 0.5) -> Metric:
    return Metric(
        name="context_relevance",
        threshold=threshold,
        instructions="Do `contexts` contain information that helps answer `input`?",
        criteria={
            "true": "At least one supplied passage contains information useful to answering the request.",
            "false": "None of the supplied passages contains information useful to answering the request.",
        },
        required_fields=("input", "contexts"),
        pass_definition="At least one retrieved passage is useful for the request; this is not context precision.",
    )
