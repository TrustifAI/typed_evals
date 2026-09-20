"""Offline CrewAI tools. Install with pip install '.[crewai]'; no API keys needed."""

import asyncio
from contextlib import asynccontextmanager

from typed_evals import GuardrailViolation, JudgeResponse
from typed_evals.adapters.crewai import guard_tool


class DemoJudge:
    """Synthetic scores demonstrate execution control, not judge accuracy."""

    model = "synthetic-crewai-demo"

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

    @guard_tool(
        policy="Only read tickets owned by Alice.",
        input="Read my support ticket.",
        contexts=["Alice owns T-42. Bob owns T-99."],
        backend=DemoJudge(),
    )
    async def read_ticket(ticket_id: str) -> str:
        """Read an owned support ticket."""
        executed.append(ticket_id)
        return "Your refund has been approved."

    # Register read_ticket in Agent(tools=[read_ticket], ...) with Crew(cache=False, ...).
    # Exercise CrewAI's native structured tool dispatch offline here.
    native_tool = read_ticket.to_structured_tool()
    for ticket_id in ("T-42", "T-99"):
        try:
            print(ticket_id, await native_tool.ainvoke({"ticket_id": ticket_id}))
        except GuardrailViolation:
            print(ticket_id, "BLOCKED")
    assert executed == ["T-42"]
    print("Actually executed:", executed)


if __name__ == "__main__":
    asyncio.run(main())
