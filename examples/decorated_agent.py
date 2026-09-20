"""Offline agent and tool decorators. Fixture scores demonstrate control flow, not accuracy."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from typed_evals import (
    EvaluationSample,
    Evaluator,
    Faithfulness,
    GuardPolicy,
    GuardrailViolation,
    JudgeResponse,
    RuntimeGuard,
    ToolProposal,
    ToolSafety,
    guarded_agent,
    guarded_by,
)


class DemoJudge:
    model = "synthetic-decorator-demo"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        proposal = state.get("proposed_tool_call", {})
        scores = {
            "tool_safety": 0.99
            if proposal.get("arguments", {}).get("ticket_id") == "T-42"
            else 0.01,
            "faithfulness": 0.05 if "90 days" in state.get("response", "") else 0.99,
        }
        return JudgeResponse(
            model=self.model,
            answers={name: {"type": "noul", "noul": scores[name]} for name in questions},
        )


backend = DemoJudge()  # Replace with JevBackend() for billable real judgments.
guard = RuntimeGuard(
    {
        "before_tool": GuardPolicy(
            Evaluator([ToolSafety(policy="Only ticket T-42 may be read.")], backend=backend)
        ),
        "after_response": GuardPolicy(
            Evaluator([Faithfulness()], backend=backend), on_fail="annotate"
        ),
    }
)
executed = []


@guarded_by(
    guard,
    before="before_tool",
    native_output=True,
    before_sample=lambda args, kwargs: EvaluationSample(
        input="Read the customer's ticket.",
        response="",
        proposed_tool_call=ToolProposal(
            name="read_ticket", arguments={"ticket_id": kwargs["ticket_id"]}
        ),
        contexts=("read_ticket(ticket_id: str) reads a ticket. The user can read T-42 only.",),
    ),
)
async def read_ticket(ticket_id: str) -> dict:
    executed.append(ticket_id)
    return {"refund_period": "30 days"}


class ExampleFrameworkAgent:
    """Representative framework interface: run returns an awaitable with a .text result."""

    def __init__(self, tools):
        self.tools = tools

    def run(self, question, *, ticket_id="T-42"):
        return self._run(question, ticket_id)

    async def _run(self, question, ticket_id):
        try:
            result = await self.tools["read_ticket"](ticket_id=ticket_id)
        except GuardrailViolation:
            return SimpleNamespace(text="I could not read that ticket.")
        assert result == {"refund_period": "30 days"}  # Tool return type remains native.
        return SimpleNamespace(
            text="The refund period is 90 days."
        )  # Deliberate unsupported claim.


@guarded_agent(
    guard,
    factory=True,
    methods=("run",),
    async_methods=("run",),
    after="after_response",
    after_sample=lambda output, args, kwargs: EvaluationSample(
        input=args[0],
        response=output.text,
        contexts=("The refund period is 30 days.",),
    ),
)
def create_agent():
    # Use your framework's constructor here, registering guarded tools as usual.
    return ExampleFrameworkAgent(tools={"read_ticket": read_ticket})


async def main():
    agent = create_agent()
    for ticket_id in ("T-42", "T-99"):
        result = await agent.run("What is my refund period?", ticket_id=ticket_id)
        print(ticket_id, result.output.text)
        print("Decisions:", [(d.checkpoint, d.action) for d in result.decisions])
        print("Hallucination suspected:", result.metadata["jev"]["hallucination_suspected"])
    print("Tools actually executed:", executed)


if __name__ == "__main__":
    asyncio.run(main())
