"""Offline Microsoft tools. Install with pip install '.[agent-framework]'."""

import asyncio
from contextlib import asynccontextmanager

from agent_framework import FunctionInvocationContext, tool

from typed_evals import GuardrailViolation, JudgeResponse
from typed_evals.adapters.agent_framework import guard_tool


class DemoJudge:
    """Synthetic scores demonstrate execution control, not judge accuracy."""

    model = "synthetic-agent-framework-demo"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        owned = state["proposed_tool_call"]["arguments"]["ticket_id"] == "T-42"
        return JudgeResponse(
            model=self.model,
            answers={name: {"type": "noul", "noul": 0.99 if owned else 0.01} for name in questions},
        )


async def main():
    executed = []

    @tool
    @guard_tool(
        policy="Only read tickets owned by the authenticated customer.",
        input=lambda ctx: ctx.kwargs["request"],
        contexts=lambda ctx: ctx.kwargs["authorization_evidence"],
        backend=DemoJudge(),
    )
    async def read_ticket(ticket_id: str, ctx: FunctionInvocationContext) -> str:
        """Read an owned support ticket."""
        executed.append(ticket_id)
        return "Your refund has been approved."

    # Agent(tools=[read_ticket], ...) injects this context from function_invocation_kwargs.
    # Supply it explicitly to exercise native FunctionTool dispatch offline.
    for ticket_id in ("T-42", "T-99"):
        context = FunctionInvocationContext(
            function=read_ticket,
            arguments={"ticket_id": ticket_id},
            metadata={"call_id": f"call-{ticket_id}"},
            kwargs={
                "request": f"Read ticket {ticket_id}",
                "authorization_evidence": [
                    "Authenticated customer Alice owns T-42. Bob owns T-99."
                ],
            },
        )
        try:
            print(ticket_id, await read_ticket.invoke(context=context, skip_parsing=True))
        except GuardrailViolation:
            print(ticket_id, "BLOCKED")
    assert executed == ["T-42"]
    print("Actually executed:", executed)


if __name__ == "__main__":
    asyncio.run(main())
