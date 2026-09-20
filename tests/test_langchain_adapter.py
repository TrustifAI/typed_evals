from __future__ import annotations

import pytest
from conftest import FakeBackend

pytest.importorskip("langchain")

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from typed_evals import GuardrailViolation
from typed_evals.adapters.langchain import guard_tool


@pytest.mark.parametrize("async_tool,async_graph", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("score", [0.99, 0.1])
async def test_real_langgraph_injection_schema_and_enforcement(async_tool, async_graph, score):
    backend = FakeBackend(
        lambda state, questions: {name: {"type": "noul", "noul": score} for name in questions}
    )
    executed = []

    def read(ticket_id: str, runtime: "ToolRuntime") -> str:  # noqa: UP037 - exercise forward refs
        """Read an owned support ticket."""
        executed.append((ticket_id, runtime.tool_call_id))
        return "Your refund has been approved."

    async def aread(ticket_id: str, runtime: ToolRuntime) -> str:
        """Read an owned support ticket."""
        return read(ticket_id, runtime)

    wrapped = guard_tool(
        policy="Read owned tickets only",
        contexts=["Alice owns T-42"],
        backend=backend,
        name="read_ticket",
    )(aread if async_tool else read)
    registered = tool("read_ticket")(wrapped)
    schema = registered.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"ticket_id"}
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([registered]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    app = graph.compile()
    state = {
        "messages": [
            HumanMessage(content="Read my ticket T-42"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_ticket",
                        "args": {"ticket_id": "T-42"},
                        "id": "call-42",
                    }
                ],
            ),
        ]
    }
    if score < 0.9:
        with pytest.raises(GuardrailViolation) as exc:
            if async_graph:
                await app.ainvoke(state)
            else:
                app.invoke(state)
        assert not executed
        assert exc.value.decision.evaluation.sample_id == "call-42"
    else:
        result = await app.ainvoke(state) if async_graph else app.invoke(state)
        assert executed == [("T-42", "call-42")]
        assert result["messages"][-1].content == "Your refund has been approved."
    judged = backend.calls[0][0]
    assert judged["input"] == "Read my ticket T-42"
    assert judged["proposed_tool_call"] == {
        "name": "read_ticket",
        "arguments": {"ticket_id": "T-42"},
    }
