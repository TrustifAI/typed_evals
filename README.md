<h1 align="center">Typed Evals</h1>

<p align="center">
  <strong>Evaluate responses. Guard actions.</strong><br>
  A Python toolkit for evaluating LLMs, RAG, and agents—with optional calibration against human labels.
</p>

<p align="center">
  <a href="https://pypi.org/project/typed-evals/"><img src="https://img.shields.io/pypi/v/typed-evals?style=flat-square&amp;color=0f766e" alt="PyPI version"></a>
  <a href="https://pypi.org/project/typed-evals/"><img src="https://img.shields.io/badge/python-3.11%2B-2563eb?style=flat-square" alt="Python 3.11 and later"></a>
  <a href="https://github.com/TrustifAI/typed_evals/actions/workflows/test.yml"><img src="https://github.com/TrustifAI/typed_evals/actions/workflows/test.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://github.com/TrustifAI/typed_evals/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-64748b?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#evaluate-agent-execution">Agent evaluation</a> ·
  <a href="#guard-tool-calls">Tool guards</a> ·
  <a href="#calibration">Calibration</a> ·
  <a href="#examples-and-guides">Examples &amp; guides</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/TrustifAI/typed_evals/main/docs/assets/readme-banner.svg" width="1200" alt="Know what passed. Decide what runs. Typed Evals takes your evidence through a metric panel and returns typed scores, thresholds, and status.">
</p>

Use **[Jev](https://typesafe.ai/)**, a System One model, to judge generated responses and recorded agent executions. Check proposed tool calls before they run. Start with a preset, then bring your own metrics, thresholds, or judge backend as your application grows.

## Quickstart

Install with **Python 3.11+** and set your [TypeSafe](https://typesafe.ai/) API key:

```bash
python -m pip install typed-evals
export TYPESAFE_API_KEY='your-key'
```

Evaluate a response against the evidence used to generate it:

```python
from typed_evals import evaluate

result = evaluate(
    input="What is the refund period?",
    response="You can request a refund within 30 days.",
    contexts=["Refunds are allowed within 30 days of purchase."],
    preset="rag",
)

print("All checks passed:", result.passed)
for name, metric in result.metrics.items():
    print(f"{name}: score={metric.score}, passed={metric.passed}")
```

That checks **faithfulness**, **answer relevancy**, and **context relevance** in one judge request. The default backend uses `jev-1.13.0` through the official `typesafe-sdk` 0.7.x; live examples make API calls using your account.

**Want to explore without an API key?** Jump to the [offline demo](#try-it-offline).

## What you can build

| Your workflow | What Typed Evals provides |
| --- | --- |
| **[Evaluate a RAG pipeline](#quickstart)** | Check whether answers address the query and stay grounded in the retrieved context. |
| **[Evaluate Agent runs](#evaluate-agent-execution)** | Judge task completion and claims about tool results against recorded execution evidence. |
| **[Guard tool calls](#guard-tool-calls)** | Evaluate proposed calls against your policy before executing the underlying function. |
| **[Define your own rubric](#write-a-custom-metric)** | Compose nine built-in metrics or write custom binary, categorical, and ordered-score checks. |
| **[Calibrate thresholds](#calibration)** | Fit per-metric curves to human pass/fail labels and inspect performance on held-out data. |

### Choose a preset

| Preset | Checks | Evidence to supply |
| --- | --- | --- |
| `"response"` · default | Answer relevancy | `input`, `response` |
| `"rag"` | Faithfulness, answer relevancy, context relevance | `input`, `response`, `contexts` |
| `"agent"` | Task completion, tool grounding | `input`, `response`, `trace`, `expected_outcome` |

Presets use a `0.5` threshold for each metric. **Supplying contexts alone does not enable RAG checks**—choose `preset="rag"` explicitly. Missing evidence and judge errors raise by default. Use custom metrics to choose different checks or thresholds.

## Evaluate agent execution

Use `preset="agent"` to check whether a completed run achieved its expected outcome and reported tool results accurately. Supply the final response, observed execution trace, and explicit success criteria:

```python
from typed_evals import evaluate

result = evaluate(
    preset="agent",
    input="Create a support ticket for the export failure.",
    response="I created ticket T-42 for your export failure.",
    expected_outcome="A new support ticket exists for the reported export failure.",
    trace=[
        {
            "name": "create_ticket",
            "arguments": {"title": "Export failure"},
            "status": "error",
            "output": {"created": False},
            "error": "Permission denied",
        }
    ],
)

for name, metric in result.metrics.items():
    print(f"{name}: score={metric.score}, passed={metric.passed}")
```

- **Task completion** checks whether execution evidence establishes `expected_outcome`. Plans, attempts, and claims of success are insufficient.
- **Tool grounding** checks whether the response's claims match the observed tool results, including failed or unverified executions.

Here, ticket creation failed while the agent claimed success. Both rubrics describe a failure; the returned scores depend on Jev's judgment.

Capture `trace` from actual executions in your integration. The preset covers completion and result reporting; step ordering, efficiency, and recovery quality need separate checks. See the [runnable example](https://github.com/TrustifAI/typed_evals/blob/main/examples/agent_evaluation.py) and the [runtime guide](https://github.com/TrustifAI/typed_evals/blob/main/docs/RUNTIME.md) for enforcing checks at agent and tool boundaries.

## Evaluate a dataset

Save your samples as `samples.jsonl`, with one JSON object per line:

```json
{"input":"Refund period?","response":"30 days.","contexts":["Refunds are allowed within 30 days of purchase."]}
```

```python
from typed_evals import evaluate

report = evaluate("samples.jsonl", preset="rag")
print(report.summary)
report.save("evaluation-report.json")
```

You can also pass a dictionary, an `EvaluationSample`, or a list/tuple of either. A single sample returns `SampleResult`; a sequence or file returns `EvaluationReport`, even for one row. JSON files contain an array of samples.

Each sample's metrics share a judge request. Batches validate inputs first, run up to **eight concurrent workers** by default, and preserve input order.

**In notebooks and async applications**, use `aevaluate` with the same dataset:

```python
from typed_evals import aevaluate

report = await aevaluate("samples.jsonl", preset="rag")
```

The synchronous API also works in notebooks, but blocks until evaluation finishes.

**From the terminal or CI:**

```bash
typed_evals evaluate samples.jsonl \
  --metrics faithfulness answer_relevancy context_relevance \
  --output evaluation-report.json \
  --fail-on-failure
```

The CLI returns a nonzero exit code for failed or unavailable evaluations when `--fail-on-failure` is set. See the [evaluation guide](https://github.com/TrustifAI/typed_evals/blob/main/docs/EVALUATION.md) for error policies, data formats, and report fields.

## Guard tool calls

Add a policy check to a LangChain tool before registering it:

```bash
python -m pip install 'typed-evals[langchain]'
```

```python
from langchain.tools import ToolRuntime, tool
from typed_evals.adapters.langchain import guard_tool


@tool
@guard_tool(
    policy="Only read tickets owned by the authenticated customer.",
    contexts=lambda runtime: runtime.context["authorization_evidence"],
)
def read_ticket(ticket_id: str, runtime: ToolRuntime) -> str:
    """Read the complete text of a support ticket."""
    return ticket_store.read_authorized(runtime.context["customer_id"], ticket_id)
```

Here, `ticket_store` is your application's ticket service. Supply the current customer ID and application-owned authorization evidence through the agent's runtime context; the [complete LangChain example](https://github.com/TrustifAI/typed_evals/blob/main/examples/langchain_guarded_tools.py) shows the wiring.

The adapter captures the latest human request, proposed arguments, tool name, and call ID. By default, it checks `ToolSafety` at a `0.9` threshold and raises `GuardrailViolation` on a failed or unavailable judgment **before the function runs**. Successful calls preserve the tool's native return value. Keep deterministic authorization inside the tool, as shown above.

### Use your framework

| Integration | Install | Import `guard_tool` from |
| --- | --- | --- |
| Plain Python | `pip install typed-evals` | `typed_evals` |
| LangChain | `pip install 'typed-evals[langchain]'` | `typed_evals.adapters.langchain` |
| CrewAI | `pip install 'typed-evals[crewai]'` | `typed_evals.adapters.crewai` |
| Microsoft Agent Framework | `pip install 'typed-evals[agent-framework]'` | `typed_evals.adapters.agent_framework` |

The plain Python helper takes explicit `input` and `contexts`, as values or callbacks. The CrewAI adapter creates a native tool; the Microsoft adapter wraps a function under `@agent_framework.tool`. See the [adapter guide](https://github.com/TrustifAI/typed_evals/blob/main/docs/ADAPTERS.md) for each framework's registration and evidence requirements, and the [runtime guide](https://github.com/TrustifAI/typed_evals/blob/main/docs/RUNTIME.md) for agent decorators and additional checkpoints.

## Make the checks yours

Choose **custom metrics instead of a preset** to control the panel and thresholds:

```python
from typed_evals import AnswerRelevancy, Faithfulness, evaluate

result = evaluate(
    input="Refund period?",
    response="30 days.",
    contexts=["Refunds are allowed within 30 days."],
    metrics=[Faithfulness(threshold=0.8), AnswerRelevancy(threshold=0.8)],
)
```

For reusable configuration, use `Evaluator(preset="rag")` or `Evaluator(metrics=[...])`.

<details>
<summary><strong>Explore the nine built-in metrics</strong></summary>

| Focus | Metrics |
| --- | --- |
| Responses and retrieval | `answer_correctness`, `answer_relevancy`, `context_relevance`, `faithfulness` |
| Recorded agent execution | `task_completion`, `tool_grounding` |
| Proposed tool calls | `tool_accuracy`, `tool_safety` |
| Content policy | `policy_compliance` |

```python
from typed_evals import list_metrics

print(list_metrics())  # No API credentials needed.
```

`PolicyCompliance` and `ToolSafety` require `policy=...` when constructed. See the [metric definitions](https://github.com/TrustifAI/typed_evals/blob/main/docs/EVALUATION.md#built-in-metrics) for required evidence and score meanings.

</details>

<a id="write-a-custom-metric"></a>

<details>
<summary><strong>Write a custom metric</strong></summary>

Define an ordered rubric from least to most desirable. Higher scores pass the threshold, so the last level should describe your best outcome:

```python
from typed_evals import Metric, evaluate

completeness = Metric(
    name="completeness",
    kind="score",
    instructions="How completely does `response` cover the details requested in `input`?",
    criteria=(
        "None of the requested details are supplied.",
        "Some details are supplied, but essential requested details are missing.",
        "All essential requested details are supplied.",
    ),
    pass_definition="All essential requested details are supplied.",
    required_fields=("input", "response"),
    threshold=0.8,
)

result = evaluate(
    input="What is the refund period, and how do I request one?",
    response="You have 30 days. Contact support with your order number.",
    metrics=[completeness],
)
```

An ordered `score` rubric produces a normalized expected level between `0` and `1`; it is not automatically a probability of human acceptance. For binary and categorical rubrics, use `kind="noul"` and `kind="choice"`. The [custom metrics example](https://github.com/TrustifAI/typed_evals/blob/main/examples/custom_metrics.py) shows all three formats.

</details>

<details>
<summary><strong>Choose another model or implement a backend</strong></summary>

Pass `backend=JevBackend(model="...")` to choose a different Jev model. Jev is the only bundled provider; custom providers implement the public `Backend` and `JudgeSession` protocols and can be passed as `backend=` to evaluation and tool guards.

An adapter supplies a model ID, an async `session()` context manager, and `async judge(state, questions)` returning a `JudgeResponse`. It must translate the TypeSafe SDK's Noul, Choice, and Score formats into provider requests and matching answers. The `typesafe-sdk` dependency remains required; the CLI uses `JevBackend`.

See the [backend contract](https://github.com/TrustifAI/typed_evals/blob/main/docs/ARCHITECTURE.md#backend-interface) and [synthetic backend example](https://github.com/TrustifAI/typed_evals/blob/main/examples/offline_demo.py).

</details>

## Calibration

**Make thresholds reflect your own acceptance criteria.** A raw judge score of `0.8` does not necessarily mean humans would pass 80% of similar examples.

With `typed-evals[calibration]`, use `EvaluationPipeline` to fit per-metric isotonic curves against representative human pass/fail labels. Inspect the held-out diagnostics, save the fitted artifact, and load it for later evaluations. Calibration is opt-in; improvements are measured on held-out data rather than assumed.

Read the [calibration walkthrough](https://github.com/TrustifAI/typed_evals/blob/main/docs/CALIBRATION.md) for fitting, validation, and saved artifacts. The included labeled datasets are synthetic examples; replace them with human-reviewed labels for your application.

`metric.score` uses the raw score unless calibration is applied. `metric.passed` compares it with the threshold and is `None` for unavailable evaluations. Judge results can be wrong; individual checks do not establish a single probability that an entire answer is true.

## Try it offline

Clone the repository to get the example scripts and datasets; they are not bundled in the installed package:

```bash
git clone https://github.com/TrustifAI/typed_evals.git
cd typed_evals
python -m pip install -e '.[calibration]'
python examples/offline_demo.py
```

This demo uses a deterministic synthetic judge: **no API key and no network calls**. It shows how a raw score, a calibrated probability, and a pass/fail decision relate.

From the same checkout, explore runtime enforcement:

```bash
python examples/runtime_guardrails.py
python examples/decorated_agent.py
```

The [CrewAI](https://github.com/TrustifAI/typed_evals/blob/main/examples/crewai_guarded_tools.py) and [Microsoft Agent Framework](https://github.com/TrustifAI/typed_evals/blob/main/examples/agent_framework_guarded_tools.py) demos also use synthetic judges; install the corresponding framework extra first.

## Examples and guides

| Start here | What you'll learn |
| --- | --- |
| [Evaluation](https://github.com/TrustifAI/typed_evals/blob/main/docs/EVALUATION.md) | Metrics, datasets, reports, custom checks, and CLI usage. |
| [Runtime guards](https://github.com/TrustifAI/typed_evals/blob/main/docs/RUNTIME.md) | Tool guards, agent decorators, and failure behavior. |
| [Framework adapters](https://github.com/TrustifAI/typed_evals/blob/main/docs/ADAPTERS.md) | LangChain, CrewAI, and Microsoft Agent Framework integration. |
| [Calibration](https://github.com/TrustifAI/typed_evals/blob/main/docs/CALIBRATION.md) | Human labels, held-out validation, and reusable artifacts. |
| [Architecture](https://github.com/TrustifAI/typed_evals/blob/main/docs/ARCHITECTURE.md) | Package structure and the backend extension contract. |
| [Runnable examples](https://github.com/TrustifAI/typed_evals/tree/main/examples) | RAG evaluation, agent traces, calibration, and guarded tools. |
| [Introductory notebook](https://github.com/TrustifAI/typed_evals/blob/main/notebooks/sample_notebook.ipynb) | An interactive introduction to evaluation. |
| [Advanced notebook](https://github.com/TrustifAI/typed_evals/blob/main/notebooks/advanced_usage.ipynb) | Custom metrics, calibration, and agent integrations. |

For notebooks, install the repository into the kernel's environment and run from the repository root. They use `python-dotenv` and make live requests with `TYPESAFE_API_KEY`. Calibration needs the `calibration` extra; the advanced agent sections also need `agent-framework`, `langchain`, `langchain-google-genai`, and Gemini credentials.

## Contribute

Found a confusing result, need an integration, or have a useful evaluation example? [Open an issue](https://github.com/TrustifAI/typed_evals/issues) or send a pull request. Include a minimal reproduction for bugs and tests for behavior changes.

From a repository checkout:

```bash
python -m pip install -e '.[calibration,dev,langchain,crewai,agent-framework]'
pytest -m "not live" --cov=typed_evals --cov-report=term-missing
ruff check .
ruff format --check .
python -m build
```

CI runs the framework integrations with fake judges. Tests for optional frameworks are skipped when those extras are absent. The separate live smoke test requires both `TYPED_EVALS_LIVE=1` and `TYPESAFE_API_KEY` and makes a billable request. Maintainers can follow the [publishing guide](https://github.com/TrustifAI/typed_evals/blob/main/docs/PUBLISHING.md) to cut a release.

---

<p align="center">
  Built by <a href="https://github.com/Aaryanverma">Aaryan Verma</a> ·
  <a href="https://github.com/TrustifAI/typed_evals/blob/main/LICENSE">MIT licensed</a><br>
  <strong>Useful for your next evaluation? Star the repo to keep it close.</strong>
</p>
