import asyncio
import inspect
from types import SimpleNamespace

import pytest
from conftest import FakeBackend

from typed_evals import (
    EvaluationSample,
    Evaluator,
    Faithfulness,
    GuardedAgent,
    GuardPolicy,
    GuardrailViolation,
    RuntimeGuard,
    guarded_agent,
    guarded_by,
)


def make_guard(score=0.9):
    backend = FakeBackend(
        lambda state, questions: {name: {"type": "noul", "noul": score} for name in questions}
    )
    return RuntimeGuard(
        {
            "before": GuardPolicy(Evaluator(backend=backend)),
            "response": GuardPolicy(
                Evaluator([Faithfulness()], backend=backend), on_fail="annotate"
            ),
            "tool": GuardPolicy(Evaluator(backend=backend)),
        }
    ), backend


def after_sample(output, args, kwargs):
    return EvaluationSample(
        input=args[0] if args else kwargs["question"],
        response=output["answer"],
        contexts=("The answer is Paris.",),
    )


def before_sample(args, kwargs):
    return EvaluationSample(input=args[0] if args else kwargs["question"], response="proposed")


@pytest.mark.parametrize("method", ["invoke", "run", "kickoff", "chat", "__call__"])
def test_existing_agent_entrypoints_are_discovered_and_native_instance_is_not_mutated(method):
    guard, backend = make_guard(0.1)
    native = {"answer": "London"}

    def invoke(self, question, *, option=None):
        assert option == "preserved"
        return native

    Agent = type("Agent", (), {method: invoke, "name": "native-agent"})
    original = Agent()
    agent = guarded_agent(guard, after="response", after_sample=after_sample)(original)
    call = agent if method == "__call__" else getattr(agent, method)
    result = call(question="Capital?", option="preserved")
    assert result.output is native
    assert result.metadata["jev"]["hallucination_suspected"] is True
    assert agent.name == "native-agent"
    assert agent.__wrapped__ is original
    assert agent.guarded_methods == (method,)
    assert getattr(original, method)("Capital?", option="preserved") is native
    assert len(backend.calls) == 1
    assert inspect.signature(call) == inspect.signature(getattr(original, method))


@pytest.mark.parametrize("method", ["ainvoke", "arun", "kickoff_async", "achat"])
async def test_native_async_agent_methods(method):
    guard, _ = make_guard()

    async def invoke(self, question):
        return {"answer": "Paris"}

    agent = type("Agent", (), {method: invoke})()
    guarded = guarded_agent(guard, after="response", after_sample=after_sample)(agent)
    assert inspect.iscoroutinefunction(getattr(guarded, method))
    result = await getattr(guarded, method)("Capital?")
    assert result.output == {"answer": "Paris"}


def test_class_decorator_guards_invocation_not_construction_and_preserves_bound_arguments():
    guard, backend = make_guard()

    @guarded_agent(guard, after="response", after_sample=after_sample)
    class Agent:
        def __init__(self, answer):
            self.answer = answer

        def run(self, question):
            return {"answer": self.answer}

    instance = Agent("Paris")
    assert not backend.calls
    assert isinstance(instance, GuardedAgent)
    assert isinstance(instance.__wrapped__, Agent.__wrapped__)
    assert instance.run("Capital?").output == {"answer": "Paris"}
    assert Agent.__name__ == "Agent"


async def test_async_factory_and_regular_methods_returning_custom_awaitables():
    guard, _ = make_guard()

    class ResponseAwaitable:
        def __await__(self):
            async def complete():
                return {"answer": "Paris"}

            return complete().__await__()

    class Agent:
        def run(self, question):
            return ResponseAwaitable()

    @guarded_agent(
        guard, after="response", after_sample=after_sample, async_methods=("run",), factory=True
    )
    async def build_agent():
        return Agent()

    agent = await build_agent()
    result = await agent.run("Capital?")
    assert result.output == {"answer": "Paris"}


def test_sync_factory_with_custom_entrypoint():
    guard, _ = make_guard()

    @guarded_agent(
        guard, after="response", after_sample=after_sample, methods=("execute",), factory=True
    )
    def build_agent(answer):
        return SimpleNamespace(execute=lambda question: {"answer": answer})

    assert build_agent("Paris").execute("Capital?").output == {"answer": "Paris"}


def test_decorated_function_and_bound_method():
    guard, _ = make_guard()

    @guarded_agent(guard, after="response", after_sample=after_sample)
    def agent(question):
        return {"answer": "Paris"}

    assert agent("Capital?").output == {"answer": "Paris"}
    native = SimpleNamespace(run=lambda question: {"answer": "Paris"})
    wrapped = guarded_agent(guard, after="response", after_sample=after_sample)(native.run)
    assert wrapped("Capital?").output == {"answer": "Paris"}


def test_entrypoint_precheck_prevents_agent_execution():
    guard, _ = make_guard(0.1)
    agent = SimpleNamespace(run=lambda question: pytest.fail("agent must not run"))
    wrapped = guarded_agent(guard, before="before", before_sample=before_sample)(agent)
    with pytest.raises(GuardrailViolation):
        wrapped.run("Capital?")


def test_internal_agent_method_calls_keep_native_outputs_without_double_evaluation():
    guard, backend = make_guard()

    class Agent:
        def run(self, question):
            # Framework implementations often compose their own entrypoints this way.
            return {"answer": self.invoke(question)["answer"]}

        def invoke(self, question):
            return {"answer": "Paris"}

    agent = guarded_agent(guard, after="response", after_sample=after_sample)(Agent())
    assert agent.run("Capital?").output == {"answer": "Paris"}
    assert len(backend.calls) == 1


async def test_native_tool_results_and_nested_decisions_reach_agent_metadata():
    guard, _ = make_guard()

    @guarded_by(guard, before="tool", before_sample=before_sample, native_output=True)
    async def lookup(question):
        return {"answer": "Paris"}

    @guarded_agent(
        guard,
        before="before",
        before_sample=before_sample,
        after="response",
        after_sample=after_sample,
    )
    async def agent(question):
        native = await lookup(question)
        assert isinstance(native, dict)
        return native

    result = await agent("Capital?")
    assert [d.checkpoint for d in result.decisions] == ["before", "tool", "response"]
    assert len({d.id for d in result.decisions}) == 3
    assert result.output == {"answer": "Paris"}


def test_framework_catching_tool_denial_cannot_hide_the_decision_or_execute_tool():
    guard, _ = make_guard(0.1)
    calls = []

    @guarded_by(guard, before="tool", before_sample=before_sample, native_output=True)
    def lookup(question):
        calls.append(question)

    @guarded_agent(guard, after="response", after_sample=after_sample)
    def agent(question):
        try:
            lookup(question)
        except GuardrailViolation:
            return {"answer": "The lookup was blocked."}
        pytest.fail("expected a tool denial")

    result = agent("Capital?")
    assert not calls
    assert [d.action for d in result.decisions] == ["block", "annotate"]
    assert [d.checkpoint for d in result.decisions] == ["tool", "response"]


async def test_concurrent_agents_and_parallel_tools_have_isolated_metadata():
    guard, _ = make_guard()

    @guarded_by(guard, before="tool", before_sample=before_sample, native_output=True)
    async def lookup(question):
        await asyncio.sleep(0)
        return {"answer": "Paris"}

    @guarded_agent(guard, after="response", after_sample=after_sample)
    async def agent(question):
        outputs = await asyncio.gather(lookup(question), lookup(question))
        return outputs[0]

    results = await asyncio.gather(agent("First"), agent("Second"))
    assert all([d.checkpoint for d in r.decisions] == ["tool", "tool", "response"] for r in results)
    left = {d.id for d in results[0].decisions}
    assert not left & {d.id for d in results[1].decisions}
    for question, result in zip(("First", "Second"), results, strict=True):
        expected_hash = before_sample((question,), {}).content_hash
        assert all(d.evaluation.sample_hash == expected_hash for d in result.decisions[:2])


async def test_cancellation_resets_decision_scope():
    guard, _ = make_guard()
    entered = asyncio.Event()

    @guarded_agent(guard, before="before", before_sample=before_sample)
    async def agent(question):
        if question == "wait":
            entered.set()
            await asyncio.Event().wait()
        return {"answer": "Paris"}

    pending = asyncio.create_task(agent("wait"))
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert len((await agent("new")).decisions) == 1


def test_unselected_known_entrypoints_and_streaming_cannot_bypass_proxy():
    guard, backend = make_guard()

    class Agent:
        def run(self, question, stream=False):
            return {"answer": "Paris"}

        def invoke(self, question):
            pytest.fail("unguarded alternate must not execute")

        def stream(self, question):
            yield "Paris"

    agent = guarded_agent(guard, after="response", after_sample=after_sample, methods=("run",))(
        Agent()
    )
    with pytest.raises(AttributeError, match="not selected"):
        agent.invoke("Q")
    with pytest.raises(AttributeError, match="buffered"):
        agent.stream("Q")
    for args, kwargs in [(("Q",), {"stream": True}), (("Q", True), {})]:
        with pytest.raises(TypeError, match="non-streaming"):
            agent.run(*args, **kwargs)
    assert not backend.calls


@pytest.mark.parametrize(
    "options",
    [
        {"methods": "run"},
        {"methods": ()},
        {"methods": ("run", "run")},
        {"methods": ("astream",)},
        {"methods": ("run",), "async_methods": ("invoke",)},
        {"methods": ("guarded_methods",)},
        {"methods": ("__init__",)},
    ],
)
def test_bad_agent_configuration_fails_at_decoration(options):
    guard, _ = make_guard()
    with pytest.raises((ValueError, TypeError)):
        guarded_agent(guard, after="response", after_sample=after_sample, **options)


async def test_installed_microsoft_agent_framework_uses_same_decorator_without_network():
    framework = pytest.importorskip("agent_framework")
    guard, _ = make_guard(0.1)

    class FakeClient(framework.BaseChatClient):
        async def _inner_get_response(self, *, messages, stream, options, **kwargs):
            return framework.ChatResponse(
                messages=[framework.Message(role="assistant", contents=["London"])]
            )

    original = framework.Agent(client=FakeClient(), name="offline-agent")
    agent = guarded_agent(
        guard,
        after="response",
        async_methods=("run",),
        after_sample=lambda output, args, kwargs: EvaluationSample(
            input=args[0],
            response=output.text,
            contexts=("The capital is Paris.",),
        ),
    )(original)
    result = await agent.run("Capital?")
    assert isinstance(result.output, framework.AgentResponse)
    assert result.output.text == "London"
    assert result.metadata["jev"]["hallucination_suspected"] is True
    assert agent.name == "offline-agent"


@pytest.mark.parametrize("score,executed", [(0.1, False), (0.9, True)])
async def test_microsoft_agent_dispatches_guarded_tools_and_retains_decisions(score, executed):
    framework = pytest.importorskip("agent_framework")
    guard, _ = make_guard(score)
    calls = []

    @guarded_by(guard, before="tool", before_sample=before_sample, native_output=True)
    async def lookup(question: str) -> dict:
        calls.append(question)
        return {"answer": "Paris"}

    class FakeClient(framework.FunctionInvocationLayer, framework.BaseChatClient):
        requests = 0

        async def _inner_get_response(self, *, messages, stream, options, **kwargs):
            self.requests += 1
            content = (
                framework.Content.from_function_call(
                    "call-1", "lookup", arguments={"question": "Capital?"}
                )
                if self.requests == 1
                else "Paris"
            )
            return framework.ChatResponse(
                messages=[framework.Message(role="assistant", contents=[content])]
            )

    client = FakeClient()
    original = framework.Agent(client=client, tools=[lookup])
    agent = guarded_agent(
        guard,
        after="response",
        methods=("run",),
        async_methods=("run",),
        after_sample=lambda output, args, kwargs: EvaluationSample(
            input=args[0],
            response=output.text,
            contexts=("The capital is Paris.",),
        ),
    )(original)
    result = await agent.run("Capital?")
    assert calls == (["Capital?"] if executed else [])
    assert client.requests == 2
    assert [d.checkpoint for d in result.decisions] == ["tool", "response"]
    assert result.decisions[0].action == ("allow" if executed else "block")
    assert isinstance(result.output, framework.AgentResponse)
