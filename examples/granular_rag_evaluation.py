"""Diagnose individual claims and retrieved passages with the existing metrics.

Run: python examples/granular_rag_evaluation.py
The default uses fixture scores, with no credentials or network requests.
Add --live to make billable Jev requests using TYPESAFE_API_KEY.
Claims must come from your application or human review; this script does not
extract claims or guarantee that they cover every assertion in an answer.
"""

import argparse
from contextlib import asynccontextmanager

from typed_evals import (
    ContextRelevance,
    EvaluationSample,
    Evaluator,
    Faithfulness,
    JevBackend,
    JudgeResponse,
)

QUERY = "What is the refund period, and is return shipping free?"
RELEVANT_PASSAGE = "Refunds are allowed within 30 days of purchase."
CLAIMS = ["Refunds are allowed within 30 days.", "Return shipping is free."]
CONTEXTS = [RELEVANT_PASSAGE] + [
    f"Product {index} comes in blue and green." for index in range(1, 10)
]


class FixtureBackend:
    """Fixture judgments for this demonstration only; not a general evaluator."""

    model = "granular-rag-fixtures"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        answers = {}
        for name in questions:
            if name == "faithfulness":
                supported = state["response"] == CLAIMS[0] and RELEVANT_PASSAGE in state["contexts"]
            else:
                supported = RELEVANT_PASSAGE in state["contexts"]
            answers[name] = {"type": "noul", "noul": 0.95 if supported else 0.05}
        return JudgeResponse(model=self.model, answers=answers)


def show_units(report, metric_name):
    threshold = report.results[0].metrics[metric_name].threshold if report.results else None
    print(f"Threshold: {threshold}")
    for row in report.results:
        metric = row.metrics[metric_name]
        print(
            f"{row.sample_id}: score={metric.score}, passed={metric.passed}, status={metric.status}"
        )

    available = [row.metrics[metric_name].passed for row in report.results]
    unavailable = sum(value is None for value in available)
    # A missing judgment must not silently disappear from a precision denominator.
    fraction = (
        sum(value is True for value in available) / len(available)
        if available and not unavailable
        else None
    )
    print(f"Fraction judged passing: {fraction}; unavailable: {unavailable}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make billable Jev API requests")
    args = parser.parse_args()
    backend = JevBackend() if args.live else FixtureBackend()
    if not args.live:
        print("Synthetic fixture judgments; these are not measured judge results.")

    # Keep all retrieved passages for each claim: support can span passages.
    claim_report = Evaluator(
        [Faithfulness(threshold=0.8)], backend=backend, errors="record"
    ).evaluate(
        [
            EvaluationSample(input=QUERY, response=claim, contexts=CONTEXTS, id=f"claim-{index}")
            for index, claim in enumerate(CLAIMS, 1)
        ]
    )
    print("\nClaim support:")
    show_units(claim_report, "faithfulness")

    # Evaluate one passage per sample, so nine irrelevant passages stay visible.
    passage_report = Evaluator(
        [ContextRelevance(threshold=0.8)], backend=backend, errors="record"
    ).evaluate(
        [
            EvaluationSample(input=QUERY, response="", contexts=[passage], id=f"passage-{index}")
            for index, passage in enumerate(CONTEXTS, 1)
        ]
    )
    print("\nPassage relevance:")
    show_units(passage_report, "context_relevance")


if __name__ == "__main__":
    main()
