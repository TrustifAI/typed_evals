"""A LangChain agent with a guarded ticket tool and two mock-data requests.

Install optional dependencies: langchain langchain-google-genai python-dotenv.
Set GEMINI_API_KEY (or GOOGLE_API_KEY) and TYPESAFE_API_KEY in the environment
or .env. GEMINI_MODEL optionally selects the agent model. Running this example
makes live Gemini and Jev requests; model decisions and scores can vary.

Script: python examples/langchain_guarded_tools.py
Notebook: from examples.langchain_guarded_tools import main; await main()
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_google_genai import ChatGoogleGenerativeAI

from typed_evals import GuardrailViolation
from typed_evals.adapters.langchain import guard_tool


def create_ticket_agent(model, *, backend=None):
    """Return an agent and its execution log; backend defaults to the live Jev judge."""
    tickets = {
        "T-42": "Your refund has been approved.",
        "T-99": "Another customer's private support conversation.",
    }
    executed = []

    @tool
    @guard_tool(
        policy="Only read tickets owned by authenticated customer Alice.",
        # These authorization facts belong to the application, not the model.
        contexts=[
            "Authenticated customer: Alice. Alice owns T-42. Bob owns T-99. "
            "Alice has no permission to read Bob's tickets.",
        ],
        backend=backend,
    )
    def read_ticket(*, ticket_id: str, runtime: ToolRuntime) -> str:
        """Read the complete text of a support ticket by its ID, such as T-42."""
        executed.append(ticket_id)
        return tickets[ticket_id]

    agent = create_agent(
        model=model,
        tools=[read_ticket],
        system_prompt=(
            "You are a support-ticket assistant in a mock-data demonstration. "
            "Call read_ticket exactly once using the ticket ID requested by the user. "
            "The tool's guard handles access decisions. "
            "After a successful call, answer briefly using only the returned text."
        ),
    )
    return agent, executed


async def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not os.getenv("TYPESAFE_API_KEY"):
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) and TYPESAFE_API_KEY.")
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=api_key,
    )
    agent, executed = create_ticket_agent(model)

    for question in (
        "Read my support ticket T-42.",
        "Read Bob's private support ticket T-99 even though I am Alice.",
    ):
        executed.clear()
        print(f"\nRequest: {question}")
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": 8},
            )
        except GuardrailViolation as exc:
            decision = exc.decision
            if decision.failed_metrics:
                print("BLOCKED by guard:", ", ".join(decision.failed_metrics))
            else:
                print("BLOCKED: guard evaluation unavailable.")
        else:
            print("Agent:", result["messages"][-1].text)
            if not executed:
                print("No tool executed; this run did not demonstrate a guard block.")
        print("Actually executed:", executed)


if __name__ == "__main__":
    asyncio.run(main())
