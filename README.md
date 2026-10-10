<h1 align="center">Typed Evals</h1>

<p align="center">
  <strong>Evaluate responses. Guard actions.</strong><br>
  A Python toolkit for checking LLM answers, evaluating agent runs, and reviewing tool calls before execution
</p>

<p align="center">
  <a href="https://pypi.org/project/typed-evals/"><img src="https://img.shields.io/pypi/v/typed-evals?style=flat-square&amp;color=0f766e" alt="PyPI version"></a>
  <a href="https://pypi.org/project/typed-evals/"><img src="https://img.shields.io/badge/python-3.11%2B-2563eb?style=flat-square" alt="Python 3.11 and later"></a>
  <a href="https://github.com/TrustifAI/typed_evals/actions/workflows/test.yml"><img src="https://github.com/TrustifAI/typed_evals/actions/workflows/test.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://github.com/TrustifAI/typed_evals/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-64748b?style=flat-square" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#choose-your-checks">Presets</a> ·
  <a href="#guard-tool-calls">Tool guards</a> ·
  <a href="#calibration">Calibration</a> ·
  <a href="#examples-and-guides">Examples &amp; guides</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/TrustifAI/typed_evals/main/docs/assets/readme-banner.svg" width="1200" alt="Know what passed. Decide what runs. Typed Evals takes your evidence through a metric panel and returns typed scores, thresholds, and status.">
</p>

Give Typed Evals a request, a response, and the evidence used to produce it. Choose
what to check, then get a score and pass/fail result for each check. You can also
evaluate recorded agent runs and apply policy checks before tools execute.

## Quickstart

Requires **Python 3.11+**. The default [Jev](https://typesafe.ai/) backend uses a
TypeSafe API key; other providers are listed [below](#choose-a-backend).

```bash
python -m pip install typed-evals
export TYPESAFE_API_KEY='your-key'
```

Check a response against retrieved evidence:

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

The RAG preset checks whether the answer is supported by the context
(**faithfulness**), addresses the question (**answer relevancy**), and has at least
one useful context passage (**context relevance**). All three share one judge
request. `passed=True` means all selected checks met their thresholds. Judge
results can be wrong; validate them against examples from your application.

This example makes an API call using your account. To explore without a key,
start with the [offline demo](#try-it-offline).

## Choose your checks

A preset selects the metrics and the evidence they need:

| Preset | Checks | Evidence to supply |
| --- | --- | --- |
| `"response"` · default | Answer relevancy | `input`, `response` |
| `"rag"` | Faithfulness, answer relevancy, context relevance | `input`, `response`, `contexts` |
| `"agent"` | Task completion, tool grounding | `input`, `response`, `trace`, `expected_outcome` |

**Choose `preset="rag"` when you want answers checked against contexts.** Passing
contexts alone leaves the default checks unchanged. Evaluation warns when supplied
contexts are unused by the active metrics.

Presets use a `0.5` threshold for each metric. Missing required evidence and judge
errors raise by default. See [metric definitions and score meanings](docs/EVALUATION.md#built-in-metrics)
for what each check measures, or [customize your checks](#make-the-checks-yours).

## Choose a backend

| Backend | Install | Credentials |
| --- | --- | --- |
| `JevBackend` · default | `pip install typed-evals` | `TYPESAFE_API_KEY` |
| `OpenAIDecisionsBackend` · [OpenAI Decisions](docs/EVALUATION.md#native-openai-decisions) | `pip install 'typed-evals[openai]'` | `OPENAI_API_KEY` |
| `SystemOneBackend` · [Clef, Laya, Strands, and compatible servers](docs/EVALUATION.md#open-and-self-hosted-system-one-models) | `pip install typed-evals` | Optional server API key |

Pass a backend to `evaluate`, `Evaluator`, or `guard_tool` to use it. See
[provider setup](docs/EVALUATION.md#choose-a-backend) for complete examples.
[Microsoft-Decision-1](docs/EVALUATION.md#microsoft-decision-1)
uses `JevBackend` with an OpenRouter model, endpoint, and API key.
`SystemOneBackend` connects to a deployed `/v1/systemone` server with an explicit
model and API root; local servers can run without a key.

**CrewAI compatibility:** the `openai` and `crewai` extras have conflicting OpenAI
SDK requirements. Install them in separate environments; combining
`typed-evals[openai,crewai]` fails dependency resolution. Details are in the
[adapter guide](docs/ADAPTERS.md#crewai).

## Evaluate agent execution

Use `preset="agent"` with an observed `trace` and an `expected_outcome` to check
whether a run achieved its goal and reported tool results accurately. Capture the
trace from actual executions in your integration.

See the [failed-ticket example](docs/EVALUATION.md#evaluate-agent-execution) for a
run that claims success after its tool fails, and [agent_evaluation.py](examples/agent_evaluation.py)
for a runnable example.

## Evaluate a dataset

Save one sample per line in `samples.jsonl`, then evaluate the file:

```python
from typed_evals import evaluate

report = evaluate("samples.jsonl", preset="rag")
print(report.summary)
report.save("evaluation-report.json")
```

You can also pass a dictionary, an `EvaluationSample`, or a sequence of samples.
Use `aevaluate` in async applications. The [dataset guide](docs/EVALUATION.md#evaluate-a-dataset)
covers file formats, return types, and async usage; [CLI and CI](docs/EVALUATION.md#cli-and-ci)
shows how to fail a build when checks fail.

## Guard tool calls

Apply a policy check before a Python function runs:

```python
from typed_evals import guard_tool


@guard_tool(
    policy="Only read tickets owned by Alice.",
    input="Read my ticket T-42.",
    contexts=["Authenticated customer: Alice. Alice owns T-42."],
)
def read_ticket(ticket_id: str) -> str:
    """Read a support ticket's complete text."""
    return ticket_store.read_authorized("Alice", ticket_id)


text = read_ticket("T-42")
```

Here, `ticket_store` is your application's ticket service. Supply current
application-owned authorization facts in `contexts`, and keep deterministic
permission checks inside the tool, as `read_authorized` does above.

By default, `ToolSafety` uses a `0.9` threshold. A failed or unavailable judgment
raises `GuardrailViolation` before the function executes. Successful calls return
the function's native output. The [runtime guide](docs/RUNTIME.md#guard-a-python-tool)
covers evidence callbacks, failure policies, and agent checkpoints.

### Use your framework

| Integration | Install | Import `guard_tool` from |
| --- | --- | --- |
| Plain Python | `pip install typed-evals` | `typed_evals` |
| LangChain | `pip install 'typed-evals[langchain]'` | `typed_evals.adapters.langchain` |
| CrewAI | `pip install 'typed-evals[crewai]'` | `typed_evals.adapters.crewai` |
| Microsoft Agent Framework | `pip install 'typed-evals[agent-framework]'` | `typed_evals.adapters.agent_framework` |

The [adapter guide](docs/ADAPTERS.md) shows registration and evidence mapping for
each framework. Install CrewAI separately from the `openai` extra because their
SDK requirements conflict. CrewAI can use the default Jev backend.

## Make the checks yours

Choose from [nine built-in metrics](docs/EVALUATION.md#built-in-metrics), select
thresholds with the `metrics` argument, or define a `Metric` with your own rubric.
For example, `Faithfulness(threshold=0.8)` requires a higher score to pass.

See [custom metrics](docs/EVALUATION.md#custom-metrics) for binary, categorical,
and ordered-score rubrics. Use `Evaluator` to reuse a configured metric panel.
[Image evidence](docs/EVALUATION.md#image-evidence) shows how to include images in
custom checks with a supported backend.

## Calibration

A raw judge score of `0.8` does not necessarily mean humans would pass 80% of
similar examples. Opt-in calibration fits metric scores to representative human
pass/fail labels using `EvaluationPipeline`.

`CalibrationConfig(enabled=True)` automatically uses **Venn–Abers below 2,000
labeled training rows per metric**, and **isotonic at 2,000 or more**. You can choose
an algorithm explicitly or change the cutoff. The cutoff is a heuristic; inspect
held-out Brier score and log loss because either algorithm can worsen performance.

The [calibration guide](docs/CALIBRATION.md) covers algorithm choices, installation,
labeling, validation, and saved artifacts. Included labels are synthetic demos;
replace them with human-reviewed labels for your application.

## Benchmark: TRIVIA+

On **645 test answers**, isotonic calibration reduced Jev's expected calibration
error (ECE) by **68.1%**, from 0.0982 to 0.0313. Hallucination detection F1 remained
similar (0.5833 → 0.5877). Read the [full benchmark](docs/BENCHMARK.md) for the
methodology, results, and limitations.

## Try it offline

The offline demos use synthetic judges and need **no API key or network calls**.
Run them from a repository checkout; examples are not included in the installed
package:

```bash
git clone https://github.com/TrustifAI/typed_evals.git
cd typed_evals
python -m pip install -e '.[calibration]'
python examples/offline_demo.py
```

Try [runtime_guardrails.py](examples/runtime_guardrails.py) for enforcement or
[decorated_agent.py](examples/decorated_agent.py) for agent integration.

## Examples and guides

| Guide | What you'll learn |
| --- | --- |
| [Evaluation](docs/EVALUATION.md) | Providers, metrics, images, datasets, reports, and CLI usage. |
| [Runtime guards](docs/RUNTIME.md) | Tool guards, agent decorators, checkpoints, and failure behavior. |
| [Framework adapters](docs/ADAPTERS.md) | LangChain, CrewAI, and Microsoft Agent Framework wiring. |
| [Calibration](docs/CALIBRATION.md) | Human labels, algorithm selection, and reusable artifacts. |
| [Architecture](docs/ARCHITECTURE.md) | Package structure and implementing a backend. |
| [FAQ](docs/FAQ.md) | Setup, notebook imports, and dependency troubleshooting. |
| [Changelog](CHANGELOG.md) | Changes and migration notes. |

Browse [runnable examples](examples/) and the [introductory](notebooks/sample_notebook.ipynb)
or [advanced](notebooks/advanced_usage.ipynb) notebooks. For individual unsupported
claims and retrieval noise, try the [claim and passage recipe](examples/granular_rag_evaluation.py).

## Contribute

[Open an issue](https://github.com/TrustifAI/typed_evals/issues) or send a pull
request. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks,
and [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

<p align="center">
  Built by <a href="https://github.com/Aaryanverma">Aaryan Verma</a> ·
  <a href="https://github.com/TrustifAI/typed_evals/blob/main/LICENSE">MIT licensed</a><br>
  <strong>Useful for your next evaluation? Star the repo to keep it close.</strong>
</p>
