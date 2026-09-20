# Evaluation API and metric semantics

Start with the [quickstart](../README.md) for direct fields, dictionaries, and presets.
The reusable APIs below remain available for explicit configuration.

## Evaluate a response

```python
from typed_evals import EvaluationSample, Evaluator, Faithfulness, AnswerRelevancy

evaluator = Evaluator(metrics=[Faithfulness(), AnswerRelevancy()])
sample = EvaluationSample(
    input="What is the refund period?",
    response="You can request a refund within 30 days.",
    contexts=["Customers may request a refund within 30 days of purchase."],
)
result = evaluator.evaluate_one(sample)

for name, metric in result.metrics.items():
    print(name, metric.raw_score, metric.score, metric.passed)
```

`evaluate_one` also works in Jupyter/VS Code notebooks. When an event loop is
already running, synchronous methods run evaluation in a worker thread and block
until it finishes. To keep the calling event loop responsive, use the async API:

```python
result = await evaluator.aevaluate_one(sample)
```

For a supplied async client (`JevBackend(client=...)`) or a custom backend with
resources tied to an event loop, use the async API in that same loop.

All metrics for a sample share one request. A batch uses a shared connection pool
and at most `max_concurrency` workers; there is no task per dataset row. Input
order is preserved. The SDK handles retries and timeouts, without a second retry
layer. Each sample is one independent judgment request, not a shared multi-sample
prompt.

```python
from typed_evals import load_dataset

report = evaluator.evaluate(load_dataset("examples/assets/rag_samples.jsonl"))
print(report.summary)
report.save("evaluation-report.json")

# To keep a notebook, FastAPI, or another running event loop responsive:
# report = await evaluator.aevaluate(samples)
```

## What the numbers mean

| Field | Meaning |
|---|---|
| `raw_score` | Noul probability of true; Choice probability mass of `pass_options`; or normalized expected Score level |
| `confidence` | Jev's distribution-concentration statistic for Choice/Score; absent for Noul |
| `calibrated_probability` | Learned estimate of the probability that a human labels this **specific metric** as passing |
| `score` | Calibrated probability when available, otherwise raw score |
| `passed` | Whether `score >= metric.threshold`; `None` for skipped/error results |

Calibration learns `raw_score → P(human metric label = 1)`. It does not train on
Jev's `confidence`, produce the generating LLM's confidence, or estimate whether
every decision made by the judge is correct. A normalized ordinal Score is not a
probability until fitted against a binary pass criterion.

For example, if examples with raw faithfulness around 0.9 have only 70% human
passes, a representative fitted curve may map that region toward 0.7. That number
is specific to the question, label rubric, model, and data distribution.

There is deliberately **no combined “probability the whole answer is true.”**
Per-metric probabilities are dependent; averaging them does not produce that
probability. `SampleResult.passed` is a policy conjunction of complete metric
results. It is `None` if any metric was skipped or failed.

Each `SampleResult` includes `tool_name`, copied from
`EvaluationSample.proposed_tool_call.name`, or `None` when no proposal is supplied.
It appears in serialized reports and runtime evaluation metadata, including
skipped or recorded-error results. Metric names such as `tool_accuracy` continue
to identify the measurement. For multiple tool calls, create one sample per
invocation and assign a unique `EvaluationSample.id` (for example, `"call_001"`);
the result preserves it as `sample_id`. Without an explicit ID, `sample_id` is
the position within that evaluation batch, so separate `evaluate_one` calls each
default to `"0"`.

## Built-in metrics

Use `list_metrics()` to discover all built-in metric names:

```python
from typed_evals import list_metrics

for name in list_metrics():
    print(name)
```

It returns a fresh, alphabetically sorted `list[str]`, including `policy_compliance`
and `tool_safety`. Their constructors require `policy=...`; listing them does not
instantiate metrics or call the judge. Custom `Metric` instances are not registered.
The function is also available from `typed_evals.metrics`.

| Metric | Evidence required | Human label 1 means |
|---|---|---|
| `AnswerRelevancy()` | `input`, `response` | The response directly addresses the request |
| `Faithfulness()` | `response`, `contexts` | Every material factual assertion is supported by the passages |
| `AnswerCorrectness()` | `input`, `response`, `reference` | Essential facts match the reference answer |
| `ContextRelevance()` | `input`, `contexts` | At least one passage is useful for answering the request |
| `TaskCompletion()` | `input`, `trace`, `expected_outcome` | Execution evidence establishes the intended outcome |
| `ToolGrounding()` | `response`, `trace` | Claims about tool results are supported by observed outputs |
| `ToolSafety(policy=...)` | `input`, `proposed_tool_call`, `contexts` | The proposed tool call complies with application policy and authorization evidence |
| `ToolAccuracy()` | `input`, `proposed_tool_call`, `contexts` | Tool selection and argument values match the supplied specifications and facts |
| `PolicyCompliance(policy=...)` | `input`, `response` | Candidate content complies with the application policy |

`ToolAccuracy()` uses `kind="choice"` with `"true"` and `"false"` options and
`pass_options=("true",)`. Its raw score is the probability assigned to `"true"`;
the default passing threshold is `0.8`.

These are explicitly defined Jev judgments, **not reproductions of Ragas or
DeepEval metric formulas**. Faithfulness is a whole-response binary judgment,
not an extracted-claim support fraction. A fact-free response may pass grounding
while failing usefulness; evaluate both dimensions. Context relevance is not
retrieval precision/recall. Reference correctness is relative to the reference,
not an independent web fact-check. Tool traces must contain externally observed
results, not just an agent's self-reported success.

## Data format

Evaluation JSONL has one sample per line. JSON arrays are also supported.

```json
{"id":"r1","input":"Refund period?","response":"30 days.","contexts":["Refunds allowed within 30 days."],"reference":"30 days."}
```

Calibration JSONL wraps the sample with **human labels keyed by metric name**:

```json
{"sample":{"id":"c1","input":"Refund period?","response":"90 days.","contexts":["Refunds allowed within 30 days."]},"labels":{"faithfulness":0,"answer_relevancy":1}}
```

Use integer `0`/`1` or booleans. Soft labels and model-generated pseudo-labels are
not the intended calibration target. Labels, metadata, IDs, and group IDs never
enter the Jev request. Each label requires the corresponding metric's evidence.
Sparse labels are allowed, provided each configured metric meets the split minima.

Use `group_id` to keep the same conversation, source document, customer case, or
near-duplicate family together. Exact duplicate sample content is rejected even
if IDs differ. Automatic splitting preserves groups; explicit train/validation
group overlap is rejected. Evaluation reusing fitting/validation content or groups
is rejected by default. `allow_calibration_overlap=True` exists for inspecting
already-seen rows; those results are not independent performance measurements.

## Custom metrics

```python
from typed_evals import Metric

completeness = Metric(
    name="completeness",
    kind="score",
    instructions="How completely does `response` cover the requested details in `input`?",
    criteria=[
        "The response supplies none of the requested details.",
        "The response supplies some requested details, with essential omissions.",
        "The response supplies every essential requested detail.",
    ],
    pass_definition="Every essential requested detail is present.",
    required_fields=("input", "response"),
    threshold=0.8,
)
```

Describe Score levels from low to high. For Choice, use a mapping of descriptions
and explicitly set `pass_options`. For Noul, ask one crisp positive criterion and
optionally supply `criteria={"true": "...", "false": "..."}`. See
[custom_metrics.py](../examples/custom_metrics.py). `pass_definition` documents the
binary target for annotators; the question and criteria are what Jev judges.
Arithmetic, counts, and exact tool status checks should remain deterministic code.

## Use with any RAG or agent framework

Map your SDK's native output into `EvaluationSample` explicitly. No LangChain,
CrewAI, Microsoft Agent Framework, or other orchestration dependency is required.

```python
from typed_evals import evaluated_by, EvaluationSample


@evaluated_by(
    evaluator,
    sample_builder=lambda output, args, kwargs: EvaluationSample(
        input=args[0],
        response=output["answer"],
        contexts=output["contexts"],
    ),
)
def ask(question):
    return your_rag.invoke(question)


evaluated = ask("Refund period?")
original_output = evaluated.output
evaluation_metadata = evaluated.evaluation.model_dump(mode="json")
```

The wrapper preserves the native output and returns evaluation alongside it.
Async functions are supported. Map agent tool calls into `ToolCall` records; see
[agent_evaluation.py](../examples/agent_evaluation.py). Streaming outputs must be
materialized before evaluation. `evaluated_by` evaluates completed outcomes.
Use the runtime APIs below to enforce checks inside the execution loop.

For multiple tool calls, let `sample_builder` return a list or tuple of
`EvaluationSample` objects, one per call. `evaluated.evaluation` then contains an
`EvaluationReport`; inspect its `.results` for individual scores and `.summary`
for aggregates. Returning one sample still produces a `SampleResult`. An empty
list produces an empty report with no judge requests.

For a sequential agent whose tools append `ToolCall` records to `trace`:

```python
from typed_evals import ToolProposal


def build_samples(output, args, kwargs):
    calls = tuple(trace)
    tool_contexts = (
        "addition_tool(a: int, b: int) returns the sum as a string.",
        "subtraction_tool(a: int, b: int) returns a minus b as a string.",
    )
    return [
        EvaluationSample(
            input=args[0] if args else kwargs["question"],
            response=output.text,
            trace=calls,
            proposed_tool_call=ToolProposal(name=call.name, arguments=call.arguments),
            contexts=tool_contexts
            + tuple(
                f"Previous tool execution: {previous.model_dump_json()}"
                for previous in calls[:index]
            ),
        )
        for index, call in enumerate(calls)
    ]


@evaluated_by(evaluator, sample_builder=build_samples)
async def ask(question):
    trace.clear()
    return await agent.run(question)


evaluated = await ask("What is (8 + 4) - 3?")
for result in evaluated.evaluation.results:
    print(result.sample_id, result.tool_name, result.metrics)
```

Use invocation-local traces when running agents concurrently. Earlier tool
outputs belong in `contexts` when later calls depend on them: `ToolAccuracy`
judges the proposal using `input` and `contexts`, without reading `trace`.

## CLI and CI

```bash
typed_evals evaluate examples/assets/rag_samples.jsonl \
  --metrics faithfulness answer_relevancy --output report.json

typed_evals calibrate examples/assets/labeled.jsonl \
  --metrics faithfulness answer_relevancy --output calibration.json

typed_evals evaluate examples/assets/test.jsonl --metrics faithfulness answer_relevancy \
  --calibration calibration.json --threshold 0.8 --output report.json --fail-on-failure
```

Exit codes: `0` completed; `1` CI gate failed (including skipped/error/empty
evaluations); `2` execution/configuration error. CLI supports built-ins; define
custom metrics in Python. By default, missing evidence and API failures raise.
Use `missing="skip"` / `--missing skip` or `errors="record"` / `--errors record`
explicitly when partial reports are appropriate. Errors and skips are excluded
from means, with separate counts.

## Sources checked

- [Official TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [TypeSafe confidence semantics](https://docs.typesafe.ai/confidence)
- [TypeSafe Score primitive](https://docs.typesafe.ai/primitives/score)
- [Anthus Jev calibration experiment](https://anth.us/blog/can-you-trust-jev-confidence/)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

The API contract is verified against `typesafe-sdk==0.7.0`. Tests establish code
behavior; real-world metric accuracy and calibration quality require your own
labeled data. Jev can misjudge long, ambiguous, or adversarial inputs. The supplied
evaluator instructions are not a proven prompt-injection defense. Model/context
token limits still apply; the framework does not silently truncate evidence.
