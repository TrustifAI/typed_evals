# Framework tool adapters

All adapters evaluate a proposed tool call before running the function, using
`ToolSafety` with threshold `0.9` by default. They preserve the function's return
value and use the shared runtime guard for blocking, timeouts, judge errors, and
argument snapshots. Sync and async functions are supported.

Register the decorated tool with your framework. Supply application-owned
authorization facts, and keep deterministic permission checks inside the tool;
model-generated arguments, retrieved content, and tool descriptions cannot grant
access. An agent entrypoint wrapper does not intercept undecorated tools.

Install only the framework you use:

```bash
python -m pip install 'typed-evals[langchain]'
python -m pip install 'typed-evals[crewai]'
python -m pip install 'typed-evals[agent-framework]'
```

From a checkout, replace `typed-evals` with `.` to install the local extra.

The `openai` and `crewai` extras cannot currently be installed together:
`typed-evals[openai,crewai]` (or `.[openai,crewai]` from a checkout) fails dependency
resolution. The supported CrewAI dependency requires OpenAI SDK `<3`, while the
native OpenAI Decisions backend requires `>=3.26.0,<4`. Install the two extras in
separate environments; the CrewAI adapter can use the default Jev backend.
The [plain Python helper](RUNTIME.md#guard-a-python-tool), LangChain adapter, and
Microsoft adapter accept `backend=OpenAIDecisionsBackend()` when installed with
the `openai` extra; see [backend setup](EVALUATION.md#choose-a-backend).

The new integrations are tested against CrewAI 1.15.22 and
`agent-framework-core` 1.19.0. CrewAI currently supports Python below 3.14;
the CI matrix covers Python 3.11–3.13. Framework dependencies remain optional.

## LangChain

Apply `typed_evals.adapters.langchain.guard_tool` beneath `@langchain.tools.tool`.
The function declares `runtime: ToolRuntime`. The adapter extracts the latest
human request and call ID; a contexts callback receives that runtime object.
The [runtime guide](RUNTIME.md#langchain-tools) shows decorator order, tool
registration, passing authenticated identity and evidence through `context`, and
handling blocked calls. See the
[complete example](../examples/langchain_guarded_tools.py).

## CrewAI

This adapter creates a native CrewAI tool, so use it in place of CrewAI's `@tool`:

```python
from typed_evals.adapters.crewai import guard_tool


def make_ticket_tool(request: str, customer_id: str, authorization_evidence: list[str]):
    @guard_tool(
        policy="Only read tickets owned by the authenticated customer.",
        input=request,
        contexts=authorization_evidence,
        name="read_ticket",
    )
    def read_ticket(ticket_id: str) -> str:
        """Read an owned support ticket."""
        return ticket_store.read_authorized(customer_id, ticket_id)

    return read_ticket
```

Here `ticket_store` is your application's ticket service. Pass the authenticated
customer ID when creating the tool, then register it:

```python
read_ticket = make_ticket_tool(request, authenticated_customer_id, authorization_evidence)
# Use read_ticket in crewai.Agent(tools=[read_ticket], ...) with Crew(cache=False, ...).
```

Create request-specific tools to capture per-run evidence.
`input` and `contexts` may also be synchronous callbacks of bound arguments,
including default values. Authorization facts must come from the application.

The adapter uses CrewAI's sanitized tool name in the judgment and preserves the
native argument schema and inferred output schema. Direct `run` and `arun` calls,
and conversion via `to_structured_tool`, retain the guard. Successful results are
not cached by this tool. Disabling crew caching also prevents an older entry in
a shared cache from bypassing the guard. No process-wide hooks are installed.

Direct invocation raises `GuardrailViolation` if judging fails or the policy
blocks the call. CrewAI's agent loop can report this as a tool error and continue
or retry; the protected function still does not run on the blocked attempt.

Run the [offline CrewAI example](../examples/crewai_guarded_tools.py) without
API credentials. CrewAI's [custom-tool documentation](https://docs.crewai.com/en/learn/create-custom-tools)
covers native registration and caching.

## Microsoft Agent Framework

Apply the adapter below Microsoft's `@tool`. Declare the injected parameter as
`FunctionInvocationContext`; it is excluded from the model's argument schema and
the tool arguments sent to the judge:

```python
from agent_framework import FunctionInvocationContext, tool

from typed_evals.adapters.agent_framework import guard_tool


@tool
@guard_tool(
    policy="Only read tickets owned by the authenticated customer.",
    input=lambda ctx: ctx.kwargs["request"],
    contexts=lambda ctx: ctx.kwargs["authorization_evidence"],
)
def read_ticket(ticket_id: str, ctx: FunctionInvocationContext) -> str:
    """Read an owned support ticket."""
    return ticket_store.read_authorized(ctx.kwargs["customer_id"], ticket_id)
```

Register the decorated tool with `agent_framework.Agent(tools=[read_ticket], ...)`.
Pass runtime evidence explicitly when running the agent:

```python
response = await agent.run(
    request,
    function_invocation_kwargs={
        "request": request,
        "customer_id": authenticated_customer_id,
        "authorization_evidence": authorization_evidence,
    },
)
```

The example's `agent`, `request`, identity, evidence, and ticket service are
application-owned values. The adapter forwards only the selected input and
contexts to the judge; other runtime kwargs and session state are excluded.
It uses `ctx.metadata["call_id"]` when present, otherwise generates a unique ID.
Use `context_parameter="invocation"` for an injected parameter named `invocation`.
If Microsoft's `@tool(name="...")` overrides the function name, pass the same
`name` to `guard_tool`.

A blocked direct `FunctionTool.invoke` raises `GuardrailViolation`. Microsoft's
automatic tool loop may convert it to a tool-error result and continue the agent.
The adapter guards the function invocation; it does not terminate the entire run.
Use `await tool.invoke(..., skip_parsing=True)` when calling a native tool directly
and wanting its raw Python result instead of Microsoft's `Content` conversion.

Run the [offline Microsoft example](../examples/agent_framework_guarded_tools.py).
Microsoft documents runtime injection and `function_invocation_kwargs` in its
[runtime-context guide](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/runtime-context).

## Shared options

All three adapters accept `policy`, `contexts`, `threshold`, `backend`, `on_fail`,
`on_error`, `timeout`, and `name`. CrewAI and Microsoft also require explicit
`input`; LangChain reads the latest human message from its runtime state.
The defaults are `on_fail="block"` and `on_error="block"`. Selecting
`"annotate"` explicitly allows the tool to run despite that outcome.

Callbacks must be synchronous. Missing or invalid evidence prevents judging and
tool execution. Argument defaults are bound before judging, and callbacks receive
copies of model arguments where applicable. Only the reviewed JSON argument
snapshot is dispatched after an allowed decision.
