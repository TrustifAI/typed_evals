"""Inspect built-in context relevance aggregation and explicit-claim support.

Run: python examples/granular_rag_evaluation.py
The default uses fake fixture verdicts, with no credentials or network requests.
Add --live to make billable Jev requests using TYPESAFE_API_KEY.
Claims must come from your application or human review; this script does not
extract claims or guarantee that they cover every assertion in an answer.
Sentence relevance is a fraction of deterministic source spans, not a truth
check or a measured accuracy claim about the fixture backend.
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
MIXED_PASSAGE = " ".join(CONTEXTS)


class FixtureBackend:
    """Fixture judgments for this demonstration only; not a general evaluator."""

    model = "granular-rag-fixtures"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        answers = {}
        for name, question in questions.items():
            if name == "faithfulness":
                supported = state["response"] == CLAIMS[0] and RELEVANT_PASSAGE in state["contexts"]
            else:
                # Explicit fake rule, not language understanding: the target unit
                # is useful exactly when it contains this known fixture sentence.
                target_index = question.instructions["target_context_index"]
                supported = RELEVANT_PASSAGE in state["contexts"][target_index]
            answers[name] = {"type": "noul", "noul": 0.95 if supported else 0.05}
        return JudgeResponse(model=self.model, answers=answers)


def show_claims(report):
    for row in report.results:
        metric = row.metrics["faithfulness"]
        print(
            f"{row.sample_id}: score={metric.score}, passed={metric.passed}, status={metric.status}"
        )


def show_relevance(report):
    for row in report.results:
        metric = row.metrics["context_relevance"]
        details = metric.details
        print(
            f"{row.sample_id}: {metric.raw_kind}={metric.raw_score}, "
            f"passed={metric.passed}, status={metric.status}"
        )
        if details is None:
            continue
        print(
            f"  {details.relevant_units}/{details.total_units} relevant; "
            f"unavailable={details.unavailable_units}; coverage={details.coverage}"
        )
        for unit in details.units:
            print(
                f"  passage={unit.passage_index}, unit={unit.unit_index}, "
                f"span=[{unit.start}:{unit.end}], P(relevant)={unit.probability}, "
                f"relevant={unit.relevant}, status={unit.status}"
            )


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
    show_claims(claim_report)

    samples = [
        EvaluationSample(input=QUERY, response="", contexts=CONTEXTS, id="ten-passages"),
        EvaluationSample(
            input=QUERY, response="", contexts=[MIXED_PASSAGE], id="one-mixed-passage"
        ),
    ]
    for granularity in ("sentence", "passage"):
        # Aggregate directly: no caller-maintained per-passage samples/denominator.
        # With fixtures, sentence mode scores both retrievals 0.1. Passage mode
        # scores ten-passages 0.1 but the one mixed passage 1.0: it is coarser.
        report = Evaluator(
            [
                ContextRelevance(
                    threshold=0.8,
                    granularity=granularity,
                    unit_threshold=0.5,
                    questions_per_call=32,
                )
            ],
            backend=backend,
            errors="record",
        ).evaluate(samples)
        print(f"\nContext relevance ({granularity} units):")
        show_relevance(report)

    print(
        "\nAny unavailable unit makes its aggregate unavailable; no failed unit "
        "is dropped from the denominator. Reports retain offsets and verdicts, "
        "not source text. Keep your original contexts to recover exact spans."
    )


if __name__ == "__main__":
    main()
