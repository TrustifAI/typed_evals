"""Run without credentials; pass --live to make billable Jev judgment requests.

The default backend supplies synthetic fixture scores to demonstrate control flow.
It is not a safety classifier and says nothing about real-world judge accuracy.
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager

from typed_evals import (
    EvaluationSample,
    Evaluator,
    Faithfulness,
    GuardPolicy,
    GuardrailViolation,
    JevBackend,
    JudgeResponse,
    RuntimeGuard,
    ToolAccuracy,
    ToolProposal,
    ToolSafety,
)


class DemoBackend:
    model = "synthetic-runtime-demo"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        proposal = state.get("proposed_tool_call", {})
        scores = {
            "tool_safety": 0.99 if proposal.get("name") == "read_ticket" else 0.01,
            "tool_accuracy": 0.99 if proposal.get("arguments") == {"ticket_id": "T-42"} else 0.01,
            "faithfulness": 0.05,  # The fixture response claims 90 days; evidence says 30.
        }
        return JudgeResponse(
            model=self.model,
            answers={
                name: {
                    "type": "choice",
                    "choice": "true" if scores[name] >= 0.5 else "false",
                    "confidence": 0.6,
                    "probabilities": {"true": scores[name], "false": 1 - scores[name]},
                }
                if question.type == "choice"
                else {"type": "noul", "noul": scores[name]}
                for name, question in questions.items()
            },
        )


async def main(live=False):
    backend = JevBackend() if live else DemoBackend()
    guard = RuntimeGuard(
        {
            "before_tool": GuardPolicy(
                Evaluator(
                    [
                        ToolSafety(
                            policy="The user may only read ticket T-42. Deleting tickets is forbidden."
                        ),
                        ToolAccuracy(),
                    ],
                    backend=backend,
                )
            ),
            "after_response": GuardPolicy(
                Evaluator([Faithfulness(threshold=0.8)], backend=backend),
                on_fail="annotate",
            ),
        }
    )
    executed = []

    async def read_ticket(ticket_id):
        executed.append(("read_ticket", ticket_id))
        return {"id": ticket_id, "refund_period": "30 days"}

    async def delete_ticket(ticket_id):
        executed.append(("delete_ticket", ticket_id))
        return {"deleted": ticket_id}

    registry = {"read_ticket": read_ticket, "delete_ticket": delete_ticket}
    contexts = (
        "read_ticket(ticket_id: str) reads a ticket. delete_ticket(ticket_id: str) deletes it.",
        "The authenticated user can read T-42. No other access or deletion is authorized.",
    )
    for name, ticket_id in (
        ("read_ticket", "T-42"),
        ("delete_ticket", "T-42"),
        ("read_ticket", "T-99"),
    ):
        sample = EvaluationSample(
            input="Read ticket T-42 to find its refund period.",
            response="",
            contexts=contexts,
            proposed_tool_call=ToolProposal(name=name, arguments={"ticket_id": ticket_id}),
        )
        try:
            result = await guard.acall_tool(registry, sample)
        except GuardrailViolation as exc:
            print(
                f"{name}({ticket_id}): {exc.decision.action}; failed={exc.decision.failed_metrics}"
            )
        else:
            print(f"{name}({ticket_id}): {result.output}")

    native_response = {"answer": "The refund period is 90 days."}
    response = await guard.arespond(
        "after_response",
        EvaluationSample(
            input="What is the refund period?",
            response=native_response["answer"],
            contexts=("The refund period is 30 days.",),
        ),
        native_response,
    )
    print("Actually executed:", executed)
    print("Native response:", response.output)
    print(json.dumps(response.metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Use the paid Jev API instead of fixture scores"
    )
    asyncio.run(main(live=parser.parse_args().live))
