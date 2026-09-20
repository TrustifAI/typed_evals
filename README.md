# typed_evals

Evaluate LLM responses, RAG datasets, and recorded agent executions using System One Models like **[Jev](https://typesafe.ai/)** as the judge. Guard tools before they execute.
Optionally calibrate individual metrics against human pass/fail labels.

**Made with ❤︎ by [Aaryan Verma](https://github.com/Aaryanverma).**

This is NOT an official [TypeSafe AI]((https://typesafe.ai/)) product.

The project, command-line tool, and Python package are all named `typed_evals`.
The Python package lives directly in the repository root.

`Requires python >= 3.11`

## Install

```bash
python -m pip install .
export TYPESAFE_API_KEY='your-key'
```

The default judge is `jev-1.13.0`, through the official `typesafe-sdk` 0.7.x.
For an offline demonstration without credentials, run `python examples/offline_demo.py`
after installing the fitting dependency with `pip install '.[calibration]'`.

## Evaluate a RAG response

```python
from typed_evals import evaluate

result = evaluate(
    input="What is the refund period?",  # your query
    response="You can request a refund within 30 days.",  # Response from your LLM
    contexts=["Refunds are allowed within 30 days of purchase."],  # context used for query
    preset="rag",
)

print(result.passed)
for name, metric in result.metrics.items():
    print(name, metric.score, metric.passed)
```

Supply your evidence and choose a preset. Credentials come from the environment;
backend configuration and metric objects are optional. Without a preset or custom
metrics, evaluation checks **answer relevancy only**, even when contexts are supplied.

| Preset | Checks | Required evidence |
|---|---|---|
| `"response"` (default) | Answer relevancy | `input`, `response` |
| `"rag"` | Faithfulness, answer relevancy, context relevance | `input`, `response`, `contexts` |
| `"agent"` | Task completion, tool grounding | `input`, `response`, `trace`, `expected_outcome` |

Presets are fixed panels using the metrics' existing defaults (threshold `0.5`
for these three panels). They never infer or drop checks based on available data.
Missing evidence and judge errors raise by default. See the
[metric definitions and score semantics](docs/EVALUATION.md#built-in-metrics)
when selecting checks and acceptance thresholds for your application.

## Evaluate a dataset

```python
report = evaluate("examples/assets/rag_samples.jsonl", preset="rag")
print(report.summary)
report.save("evaluation-report.json")
```

`evaluate` also accepts one dictionary, an `EvaluationSample`, or a list/tuple
of dictionaries and samples. Direct fields or one sample return `SampleResult`;
a sequence or file always returns `EvaluationReport`, including empty and
single-row datasets. JSON files contain arrays; JSONL files contain one row per line:

```json
{"input":"Refund period?","response":"30 days.","contexts":["Refunds allowed within 30 days."]}
```

All rows are validated before judging starts. Each sample's metrics share one
request; batches use up to eight concurrent workers by default and preserve input order.

In notebooks or async applications, use `aevaluate` to keep the event loop responsive:

```python
from typed_evals import aevaluate

report = await aevaluate("examples/assets/rag_samples.jsonl", preset="rag")
```

The synchronous API also works in notebooks, but blocks until evaluation finishes.

## Guard a LangChain tool

Install `pip install '.[langchain]'`, then decorate the tool before registering it:

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

Here `ticket_store` is your application's ticket service. Supply current,
application-owned authorization evidence in the agent's runtime context.
The adapter captures the latest human text, tool name, arguments, and call ID.
It uses `ToolSafety` with threshold `0.9` and raises `GuardrailViolation` on a
failed or unavailable judgment before the function executes. Native outputs
are preserved. The tool still enforces deterministic authorization when it runs.

See the complete [LangChain example](examples/langchain_guarded_tools.py), including
agent construction and handling blocked calls. For plain Python tools, import
`guard_tool` from `typed_evals` and supply `input` and `contexts` as values or callbacks
of bound arguments. The [runtime guide](docs/RUNTIME.md) covers both helpers,
custom metrics, calibrated guards, and additional checkpoints.

## CrewAI and Microsoft Agent Framework

Framework integrations are optional:

| Framework | Install from this repository | Adapter |
|---|---|---|
| LangChain | `pip install '.[langchain]'` | `typed_evals.adapters.langchain.guard_tool` |
| CrewAI | `pip install '.[crewai]'` | `typed_evals.adapters.crewai.guard_tool` |
| Microsoft Agent Framework | `pip install '.[agent-framework]'` | `typed_evals.adapters.agent_framework.guard_tool` |

The CrewAI decorator creates a native tool ready for `Agent(tools=[...])`.
The Microsoft adapter wraps a function under `@agent_framework.tool` and reads
explicitly selected evidence from its injected `FunctionInvocationContext`.
Both use the same runtime guard and default tool-safety threshold of `0.9`.
See the [adapter guide](docs/ADAPTERS.md) for registration, runtime evidence, and
framework error-handling details. Try the offline native-tool examples:

```bash
python examples/crewai_guarded_tools.py
python examples/agent_framework_guarded_tools.py
```

## Available metrics

List all built-in metric names without API credentials:

```python
from typed_evals import list_metrics

print(list_metrics())
# ['answer_correctness', 'answer_relevancy', 'context_relevance',
#  'faithfulness', 'policy_compliance', 'task_completion',
#  'tool_accuracy', 'tool_grounding', 'tool_safety']
```

The function returns a new, alphabetically sorted list. It includes
`PolicyCompliance` and `ToolSafety`, which require `policy=...` when constructed.
Custom `Metric` instances are not registered in this list. See the
[metric definitions](docs/EVALUATION.md#built-in-metrics) for constructors and required evidence.

## Customize when needed

Choose custom metrics **instead of** a preset to control the panel and thresholds:

```python
from typed_evals import Faithfulness, AnswerRelevancy

result = evaluate(
    input="Refund period?",
    response="30 days.",
    contexts=["Refunds are allowed within 30 days."],
    metrics=[Faithfulness(threshold=0.8), AnswerRelevancy(threshold=0.8)],
)
```

Create a custom metric and use it for evaluation:

```python
from typed_evals import Faithfulness, AnswerRelevancy, Metric

# create a new metric
WeirdMetric = Metric(
    name="weirdness",
    kind="score",  # noul, choice, score (from Jev)
    instructions="Assess how weird the response is.",
    criteria=[
        "The response is completely normal and expected.",
        "The response is somewhat unusual but still understandable.",
        "The response is very weird and unexpected.",
    ],
    pass_definition="The response is not weird.",
    required_fields=("input", "response"),
    threshold=0.8,
)

# add it to metrics
result = evaluate(
    input="Refund period?",
    response="30 days.",
    contexts=["Refunds are allowed within 30 days."],
    metrics=[
        Faithfulness(threshold=0.8),
        AnswerRelevancy(threshold=0.8),
        WeirdMetric,
    ],
)
```


For reusable configuration, use `Evaluator(preset="rag")` or `Evaluator(metrics=[...])`.
Existing `EvaluationSample`, `Evaluator`, `EvaluationPipeline`, and runtime APIs
remain available. Pass `backend=JevBackend(model="...")` to select a different Jev model.

## Use another judge backend

The Python API supports custom judge backends through the public `Backend` and
`JudgeSession` protocols. Jev is the only bundled provider today, but a future
provider can be integrated by implementing an adapter and passing it as `backend=`
to `evaluate`, `aevaluate`, `Evaluator`, or `EvaluationPipeline`. Runtime guards
can use the same adapter through their evaluator or `guard_tool(backend=...)`.

A backend supplies:

- `model: str`, identifying the configured judge model.
- `session()`, an async context manager yielding a judge session.
- `async judge(state, questions)` on that session, returning a `JudgeResponse`
  with the actual model ID, answers keyed by metric name, and optional usage.

Once your adapter is implemented, use it directly:

```python
from typed_evals import evaluate
from your_app.backends import YourBackend  # Your provider adapter.

result = evaluate(
    input="What is the refund period?",
    response="You can request a refund within 30 days.",
    backend=YourBackend(),
)
```

The current question and answer contract uses the TypeSafe SDK's Noul, Choice,
and Score formats. Your adapter must translate the rubrics into provider requests
and return answers matching the selected metrics' formats and score semantics;
the `typesafe-sdk` dependency is still required. The CLI currently uses `JevBackend`.
See the [backend interface](docs/ARCHITECTURE.md#backend-interface) and the
[synthetic backend example](examples/offline_demo.py) for an existing implementation.

## Calibration

A raw judge score of `0.8` does not necessarily mean humans would pass 80% of
similar examples. Calibration uses representative human labels to align each
metric's scores with observed pass rates, helping you set meaningful thresholds
for your use case.

Calibration is off by default. Use `EvaluationPipeline` to fit or load per-metric
curves against representative human labels; see the [calibration guide](docs/CALIBRATION.md).
`score` is the raw metric score unless calibration is applied. `passed` compares
the score with the metric threshold; it is `None` for unavailable evaluations.
Jev judgments can be wrong, and the checks do not establish a single probability
that an entire answer is true.

## Documentation, notebooks, and examples

| Folder | Contents |
|---|---|
| [`typed_evals/`](typed_evals/) | Main Python package: evaluation, metrics, calibration, runtime guards, adapters, and CLI. |
| [`docs/`](docs/) | Evaluation, runtime, calibration, and architecture guides. |
| [`notebooks/`](notebooks/) | Interactive walkthroughs for response, RAG, and agent evaluation. |
| [`examples/`](examples/) | Python scripts and sample datasets for evaluation and tool guards. |
| [`tests/`](tests/) | Unit, integration, and optional live API tests. |

Start with the guide for your use case:

- [Evaluation guide](docs/EVALUATION.md): metrics, data formats, custom checks, wrappers, CLI.
- [Runtime guide](docs/RUNTIME.md): tool guards, agent decorators, policy and failure semantics.
- [Calibration guide](docs/CALIBRATION.md): fitting, held-out validation, saved artifacts.
- [Adapter guide](docs/ADAPTERS.md): LangChain, CrewAI, and Microsoft Agent Framework tools.
- [Architecture](docs/ARCHITECTURE.md): implementation boundaries.

The Python package uses lowercase subpackages grouped by responsibility:

```text
typed_evals/
├── adapters/       # Framework integrations
├── backends/       # Judge SDK and backend protocols
├── calibration/    # Fitting, diagnostics, and saved calibration
├── data/           # Validated models and dataset loading
├── evaluation/     # Evaluator, pipeline, and result decorators
├── metrics/        # Base metric, RAG/agent rubrics, registry, and presets
└── runtime/        # Guards, agent wrappers, and tool decorators
```

Public imports such as `from typed_evals import Evaluator, Faithfulness` stay the
same. For direct module imports, use the paths in the [architecture guide](docs/ARCHITECTURE.md#modules).

For interactive usage, open the [introductory notebook](notebooks/sample_notebook.ipynb)
or the [advanced notebook](notebooks/advanced_usage.ipynb), which covers custom
metrics, calibration, and agent integrations. Install this repository into the
notebook kernel's environment and use the repository root as the working directory
so the relative dataset paths resolve. The notebooks use `python-dotenv` and make
live requests with `TYPESAFE_API_KEY`; calibration also needs `.[calibration]`.
The advanced agent sections additionally use `agent-framework`, `langchain`,
`langchain-google-genai`, and Gemini credentials.

Examples you can run without API credentials after installing `.[calibration]`:

```bash
python examples/offline_demo.py
python examples/runtime_guardrails.py
python examples/decorated_agent.py
```

For live evaluation, see [RAG evaluation](examples/rag_evaluation.py),
[recorded agent evaluation](examples/agent_evaluation.py), and the
[calibrated pipeline](examples/calibrated_pipeline.py). The
[custom metrics](examples/custom_metrics.py) example defines reusable rubrics, and
the [LangChain tool guard](examples/langchain_guarded_tools.py) example demonstrates
guarding an agent's tool calls.

## Development

```bash
python -m pip install -e '.[calibration,dev,langchain,crewai,agent-framework]'
pytest --cov=typed_evals --cov-report=term-missing
ruff check .
ruff format --check .
python -m build
```

Tests cover SDK serialization, retries, evaluation, calibration, runtime enforcement,
and native LangChain, CrewAI, and Microsoft Agent Framework tool dispatch with fake judges.
CI installs all framework extras so integration tests run. Without an extra, that
framework's integration tests are skipped. The optional live smoke test
makes a billable API call only with both `TYPED_EVALS_LIVE=1` and `TYPESAFE_API_KEY` set.
