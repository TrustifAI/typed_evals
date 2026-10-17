# Evaluation API and metric semantics

Start with the [quickstart](../README.md#quickstart) for a first RAG evaluation.
This guide covers direct calls, reusable configuration, providers, and datasets.

## When to use this

| Use Typed Evals when you need to… | Choose another check when you need… |
| --- | --- |
| Compare responses against supplied evidence or a reference answer | Independent fact verification without authoritative evidence |
| Judge recorded agent outcomes against explicit success criteria | Proof that an action happened without an observed execution trace |
| Apply an application-specific rubric and threshold | Exact arithmetic, schema validation, permissions, or tool status checks |
| Measure a judge against representative human labels | A universal accuracy guarantee or a combined probability that an answer is true |

Without `metrics` or `preset`, evaluation checks answer relevancy only. The
explicit `"response"` preset adds answer completeness. Choose `"rag"`
explicitly to check retrieved contexts; supplying evidence does not select metrics.
If supplied contexts are unused by active metrics, evaluation emits one aggregate
`UserWarning` per batch before provider requests. Skipped metrics do not consume evidence.
Faithfulness judges a whole response, context relevance measures the fraction of
useful source units, and context sufficiency asks whether the passages cover the entire request.
Use [individual claims and passages](#diagnose-individual-claims-and-passages)
when you need finer diagnosis.

| Find the contract for… | Read |
| --- | --- |
| Calls, concurrency, and async clients | [Evaluate a response](#evaluate-a-response) |
| Recorded agent outcomes and tool-result claims | [Agent evaluation](#evaluate-agent-execution) |
| Provider configuration or images | [Backends](#choose-a-backend) · [Image evidence](#image-evidence) |
| Scores, thresholds, and built-in rubrics | [Score meanings](#what-the-numbers-mean) · [Metrics](#built-in-metrics) |
| Datasets and your own rubrics | [Data format](#data-format) · [Custom metrics](#custom-metrics) |
| Integration and command-line gates | [Framework mapping](#use-with-any-rag-or-agent-framework) · [CLI and CI](#cli-and-ci) |

For troubleshooting, see the [FAQ](FAQ.md). Package migration notes are in the
[changelog](../CHANGELOG.md#migration-notes).

## Evaluate a response

Use `evaluate(...)` for direct sample fields or a sample dictionary. Use an
`Evaluator` when you want to reuse a metric panel and backend across calls.
Choose `metrics=[...]` or `preset=...`; passing both raises an error. Custom
panels can set each metric's threshold independently:

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

For a supplied async client (`JevBackend(client=...)`,
`OpenAIDecisionsBackend(client=...)`, or `SystemOneBackend(client=...)`) or a custom
backend with resources tied to an event loop, use the async API in that same loop.

Single-judgment metrics for a sample share one request. `ContextRelevance` uses
separate requests, each containing at most `questions_per_call` source-unit
questions (32 by default). Its requests contain no generated response or reference
answer. An empty context set needs no unit request. A batch validates inputs before
judge requests and uses one shared connection pool with at most `max_concurrency`
sample workers (eight by default). Chunks run sequentially within each worker;
concurrency does not multiply by the number of units. There is no task per dataset
row or unit, and input order is preserved. Jev and Decisions delegate retries and
timeouts to their SDKs; System One manages retries around its HTTP requests.
Different samples never share a judge prompt.

For `N` source units, context relevance adds `ceil(N / questions_per_call)` judge
requests, plus one shared request when other active single-judgment metrics are
selected. With the default RAG preset, ten units therefore need two requests;
100 units need five. These are logical requests before provider retries.

`questions_per_call` limits the number of unit questions, not request tokens.
Each chunk repeats the complete source passages needed to disambiguate its units.
A long passage spread across many chunks can therefore multiply input-token cost
or exceed the provider's context limit. There is no silent truncation; provider
failures follow the configured error policy. Passage granularity reduces unit
requests but gives a coarser measurement.

### Evaluate a dataset

Save samples as `samples.jsonl`, with one JSON object per line:

```json
{"input":"Refund period?","response":"30 days.","contexts":["Refunds are allowed within 30 days of purchase."]}
```

```python
from typed_evals import evaluate

report = evaluate("samples.jsonl", preset="rag")
print(report.summary)
report.save("evaluation-report.json")
```

You can pass an `EvaluationSample`, a dictionary, or a list/tuple of either.
A single sample returns `SampleResult`; sequences and files return
`EvaluationReport`, even for one row. JSON files contain an array of samples.
For a reusable evaluator, load rows with `load_dataset` and pass them to
`evaluator.evaluate(...)`:

```python
from typed_evals import load_dataset

report = evaluator.evaluate(load_dataset("examples/assets/rag_samples.jsonl"))
```

Use `aevaluate` or `evaluator.aevaluate(samples)` in an async application:

```python
from typed_evals import aevaluate

report = await aevaluate("samples.jsonl", preset="rag")
```

See [data formats](#data-format) for IDs, grouping, and calibration labels, and
[CLI and CI](#cli-and-ci) for running the same checks from the terminal.

## Evaluate agent execution

Use `preset="agent"` to check whether a completed run achieved its expected
outcome and reported tool results accurately. Supply the final response,
observed execution trace, and explicit success criteria:

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

`TaskCompletion()` checks whether execution evidence establishes
`expected_outcome`; plans, attempts, and claims of success are insufficient.
`ToolGrounding()` checks the response's claims against observed tool outputs,
including failed or unverified executions. Here, ticket creation failed while
the agent claimed success. Both rubrics describe a failure; the returned scores
depend on the selected judge's judgment.

Capture `trace` from actual executions in your integration. The preset covers
completion, result reporting, response relevance, and response completeness;
step ordering, efficiency, and recovery quality
need separate checks. See the [runnable agent example](../examples/agent_evaluation.py)
and [runtime guide](RUNTIME.md) for checks at agent and tool boundaries.

## Choose a backend

Jev (`jev-1.14.0`) is the default, using the official `typesafe-sdk` `>=0.7.5,<0.8`.
Install the core package and set a [TypeSafe](https://typesafe.ai/) API key:

```bash
python -m pip install typed-evals
export TYPESAFE_API_KEY='your-key'
```

Pass `backend=JevBackend(model="...")` to choose another Jev model. Hosted
TypeSafe-compatible models use the [configuration below](#hosted-typesafe-compatible-models).
For open and self-hosted models, use
[`SystemOneBackend`](#open-and-self-hosted-system-one-models) with a deployed endpoint.
For Microsoft-Decision-1 hosted in Microsoft Foundry, use the
[Foundry configuration](#microsoft-foundry) with `SystemOneBackend`.
Custom providers extend the public `BaseBackend`, implement `session()` yielding a
`JudgeSession`, and translate questions and answers for their provider. The base
supplies configuration validation and text-only defaults. Existing custom backends
that implement the structural `Backend` protocol remain supported;
see the [backend contract](ARCHITECTURE.md#backend-interface) and
[synthetic backend example](../examples/offline_demo.py).

### OpenAI Decisions API

Install the optional official SDK and set its normal environment variable:

```bash
python -m pip install 'typed-evals[openai]'
export OPENAI_API_KEY='your-key'
```

```python
from typed_evals import OpenAIDecisionsBackend, evaluate

backend = OpenAIDecisionsBackend(model="gpt-6-luna", timeout=30.0, max_retries=2)
result = evaluate(
    input="What is the refund period?",
    response="You can request a refund within 30 days.",
    contexts=["Refunds are allowed within 30 days of purchase."],
    preset="rag",
    backend=backend,
)
print(result.passed)
```

This calls `POST /v1/decisions` through `AsyncOpenAI.decisions.create` with SDK
`>=3.26.0,<4`. The [official Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
documents the endpoint, typed answers, and supported models. Single-judgment
metrics share a request; context relevance uses separate bounded unit requests.
Samples support text, structured tool evidence, and images.

The default Decisions model is `gpt-6-luna`. An explicit `api_key=` is optional;
the official SDK otherwise uses `OPENAI_API_KEY`. One owned async client is
shared by workers for a batch and closed after success, errors, or cancellation.
Injected `AsyncOpenAI` clients remain open with their own timeout/retry settings.
Evaluate asynchronously in the supplied client's owning event loop; the SDK
handles transport retries.
OpenAI is imported lazily, so Jev and custom providers need no OpenAI installation.
See the [runnable text example](../examples/openai_decisions.py) and
[image example](../examples/openai_decisions_images.py).

Decisions maps Noul to predicates, Choice to string-valued choices, and Score
to ordered levels. See [score meanings](#what-the-numbers-mean) for their raw
scores and separate provider confidence, and [error handling](#cli-and-ci) for
per-metric refusals and partial reports. Runtime guards apply their configured
error policy. If you also use CrewAI, follow the
[separate environment requirement](ADAPTERS.md#crewai).

### Open and self-hosted System One models

`SystemOneBackend` connects to any server implementing the Jev/SystemOne
`POST /v1/systemone` request and response contract. It supports text and structured
JSON evidence, with `noul`, `choice`, and `score` questions. Laya and Strands
Decider provide compatible servers in their official serving tools.

Install `typed-evals`, start your chosen server separately, then supply its model
identifier and API root:

```python
from typed_evals import SystemOneBackend, evaluate

backend = SystemOneBackend(
    model="convaiinnovations/laya-typed-decisions",
    base_url="http://localhost:8000",
    timeout=60.0,
    max_retries=2,
)
result = evaluate(
    input="What is the refund period?",
    response="You can request a refund within 30 days.",
    contexts=["Refunds are allowed within 30 days of purchase."],
    preset="rag",
    backend=backend,
)
print(result.passed)
```

`base_url` is the API root, without `/v1/systemone`; a prefix such as
`https://example.com/my-server` is preserved. `model` is passed to the deployed
server. This adapter does not download weights or turn a Hugging Face model repo
into an inference endpoint. A custom Hugging Face deployment works when it exposes
the same System One route.

| Model | Serving setup | Backend configuration |
| --- | --- | --- |
| [Laya](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md) | `pip install 'laya[serve]'`, then `laya-serve` | `model="english"`, `"convaiinnovations/laya-multilingual"`, or `"convaiinnovations/laya-typed-decisions"`, `base_url="http://localhost:8000"` |
| [Strands Decider](https://github.com/strands-labs/strands-decider) | `pip install strands-decider`, then `strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000` | `model="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"`, `base_url="http://localhost:8000"` |

Follow the linked serving guides for supported devices and current package
versions. These examples identify the server protocol; its context and option
limits still apply.
`SystemOneBackend` defaults to `image_mode="disabled"`. To evaluate images with
a vision-capable model on a server accepting SGLang's image extension, explicitly
set `image_mode="native"`; see [image evidence](#image-evidence) and the
[runnable image example](../examples/systemone_images.py). Basic System One
compatibility alone does not imply image support. Keep images disabled for Laya,
Strands, and other deployments without that model and endpoint capability.

Keys are optional: `api_key=None` sends no bearer authorization and does not read
`TYPESAFE_API_KEY` or `TYPESAFE_BASE_URL`. For a protected endpoint, pass
`api_key=os.environ["MY_SERVER_API_KEY"]`. Laya requires a bearer token when its
server sets `LAYA_API_KEY`; the Strands local server binds to `127.0.0.1` without
authentication. See the linked serving documentation for deployment settings.

Laya's `model="convaiinnovations/laya"` lets its router choose a checkpoint;
`english`, `multilingual`, and `typed-decisions` pin bundled checkpoints. For
calibrated evaluations, keep the served checkpoint revision and routing settings
fixed, and refit after changing them. Reported model aliases can remain unchanged
when underlying weights change. Provider confidence formulas also differ; raw
metric signals continue to come from the typed probabilities or expected score.

The CLI uses the same backend:

```bash
typed_evals evaluate examples/assets/rag_samples.jsonl --backend systemone \
  --model convaiinnovations/laya-typed-decisions --base-url http://localhost:8000 \
  --metrics faithfulness answer_relevancy --output laya-report.json

# Protected server: reads the key from this environment variable.
typed_evals evaluate examples/assets/rag_samples.jsonl --backend systemone \
  --model english --base-url http://localhost:8000 \
  --api-key-env LAYA_API_KEY --metrics answer_relevancy --output laya-report.json
```

`--backend systemone` requires both `--model` and `--base-url`; omit
`--api-key-env` for a keyless server. The same options work with `calibrate`.
See [systemone_decisions.py](../examples/systemone_decisions.py) for a runnable
Python example with configurable model, URL, and optional key environment variable.

### Microsoft Foundry

Use Foundry-hosted **Microsoft-Decision-1 as the evaluation judge** with
`SystemOneBackend`. Deploy the model from the Foundry catalog, then copy the
resource endpoint, deployment name, and API key from the deployment details.
See [Microsoft's deployment and API guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision).

The request's `model` is your **deployment name**. Set `AZURE_ENDPOINT` to the
resource root, such as `https://your-resource.services.ai.azure.com`, without
`/api/projects/...`, `/openai/v1`, or the decision route. The backend configuration
adds `/providers/microsoft`, and the backend appends `/v1/systemone`.

Install `python-dotenv` if you want to load credentials from `.env`:

```bash
python -m pip install typed-evals python-dotenv
```

```dotenv
AZURE_ENDPOINT=https://your-resource.services.ai.azure.com
DEPLOYMENT_NAME=your-decision-1-deployment
AZURE_API_KEY=your-foundry-key
```

In a notebook, run:

```python
import os

import httpx2
from dotenv import load_dotenv

from typed_evals import SystemOneBackend, aevaluate

load_dotenv()

async with httpx2.AsyncClient(
    headers={"api-key": os.environ["AZURE_API_KEY"]},
    timeout=60.0,
) as client:
    backend = SystemOneBackend(
        model=os.environ["DEPLOYMENT_NAME"],
        base_url=os.environ["AZURE_ENDPOINT"].rstrip("/") + "/providers/microsoft",
        client=client,
        api_key=None,
        image_mode="disabled",
    )
    result = await aevaluate(
        input="What is the refund period?",
        response="You can request a refund within 30 days.",
        contexts=["Refunds are allowed within 30 days of purchase."],
        preset="rag",
        backend=backend,
    )

print("Overall passed:", result.passed)
for name, metric in result.metrics.items():
    print(f"{name}: score={metric.score}, passed={metric.passed}")
```

In a Python script, place the asynchronous code inside `async def main()` and
call `asyncio.run(main())`. Keep the client and evaluation in the same event loop;
the injected client belongs to the caller and must remain open during evaluation.

Azure resource keys require the `api-key` header shown above. The backend's
`api_key` parameter sends `Authorization: Bearer ...`, so leave it as `None`
when authenticating with a resource key. The client controls the timeout for
injected clients.

Foundry also accepts Microsoft Entra ID bearer tokens for
`https://cognitiveservices.azure.com/.default`. For a short evaluation, pass a
current access token as `SystemOneBackend(api_key=token, ...)` instead of the
API-key client above. For long-running evaluations, use an injected client with
authentication that refreshes tokens. No `azure-ai-inference` client is required
for this System One route. Keep images disabled for Decision-1.

### Hosted TypeSafe-compatible models

Use `JevBackend` with your server's model ID, API root, and API key:

```python
import os

from typed_evals import Evaluator, JevBackend

evaluator = Evaluator(
    preset="rag",
    backend=JevBackend(
        model="my-judge-model",
        base_url="https://judge.example.com",
        api_key=os.environ["MY_JUDGE_API_KEY"],
        timeout=30.0,
        max_retries=2,
    ),
)
result = evaluator.evaluate_one(sample)
```

The SDK sends `POST https://judge.example.com/v1/systemone` with bearer-token
authentication. `base_url` is the API root; the SDK appends `/v1/systemone`,
including after any path prefix in your root. The server must accept the
TypeSafe `state`, `questions`, and `model` payload and return TypeSafe-compatible
Noul, Choice, and Score answers, plus the response model and usage. An
OpenAI-compatible chat endpoint requires an adapter implementing `Backend`.

If `base_url` or `api_key` is omitted, the SDK reads `TYPESAFE_BASE_URL` or
`TYPESAFE_API_KEY`. An explicit value takes precedence. With no base URL set,
the SDK uses the official TypeSafe endpoint. These environment variables also
configure the default Jev backend and CLI.

For additional SDK configuration, pass your own client:

```python
from typesafe_sdk import AsyncTypeSafeClient

async with AsyncTypeSafeClient(
    model="my-judge-model",
    base_url="https://judge.example.com",
    api_key=os.environ["MY_JUDGE_API_KEY"],
) as client:
    evaluator = Evaluator(preset="rag", backend=JevBackend(model="my-judge-model", client=client))
    result = await evaluator.aevaluate_one(sample)
```

An injected client owns its base URL, credentials, retries, timeout, and
lifecycle; `JevBackend` passes its `model` to each judgment call. Use it inside
the client's owning event loop. The evaluator leaves that client open.

#### Microsoft-Decision-1

[Microsoft Decision-1 on OpenRouter](https://openrouter.ai/microsoft/microsoft-decision-1)
uses `JevBackend` through
[OpenRouter's TypeSafe-compatible System One endpoint](https://openrouter.ai/docs/guides/community/typesafe-sdk).
For an Azure-hosted deployment, use the [Microsoft Foundry setup](#microsoft-foundry).
Set `OPENROUTER_API_KEY` in your environment:

```bash
python -m pip install typed-evals python-dotenv
export OPENROUTER_API_KEY='your-key'
```

```python
import os

from dotenv import load_dotenv

from typed_evals import JevBackend, evaluate

load_dotenv()
backend = JevBackend(
    model="microsoft/microsoft-decision-1",
    base_url="https://openrouter.ai/api",
    api_key=os.environ["OPENROUTER_API_KEY"],
    image_mode="disabled",
)
result = evaluate(
    input="What is the refund period?",
    response="You can request a refund within 30 days.",
    preset="response",
    backend=backend,
)
print(result.passed)
```

This sends `POST https://openrouter.ai/api/v1/systemone`; supply
`https://openrouter.ai/api` as the base URL. The model is served through
`JevBackend`'s existing TypeSafe request and response contract. The core package
already includes the provider SDK needed for this configuration. Keep
`image_mode="disabled"` for Decision-1, which does not support image evidence.
Models using this same protocol share `JevBackend`; a model with a different
protocol can use a new `BaseBackend` subclass.

The `"response"` preset checks answer relevancy and completeness. The same backend can be passed
to `Evaluator`, `EvaluationPipeline`, and runtime guards; select `"rag"` or
`"agent"` and provide their required evidence when evaluating those workflows.
You can also put `OPENROUTER_API_KEY=your-key` in `.env`; call `load_dotenv()`
before reading it. See the [runnable example](../examples/openrouter_decisions.py) and
[introductory notebook](../notebooks/sample_notebook.ipynb).

## Image evidence

`EvaluationSample.images` is a tuple of provider-independent `ImageInput` objects;
it defaults to empty. Direct `evaluate(images=[...], ...)`, dictionaries, and
JSON/JSONL datasets accept the same image evidence. Select `"images"` in a custom
metric's `required_fields` to include them in the judge request:

```python
from typed_evals import ImageInput, JevBackend, Metric, evaluate

visual_grounding = Metric(
    name="visual_grounding",
    kind="noul",
    instructions=(
        "Does response answer input with visual claims supported by the supplied images?"
    ),
    pass_definition="Every material visual claim is supported by the supplied images.",
    required_fields=("input", "response", "images"),
    threshold=0.8,
)
result = evaluate(
    input="What total is shown on the receipt?",
    response="The total is $42.00.",
    images=[ImageInput.from_file("receipt.png", label="receipt")],
    metrics=[visual_grounding],
    backend=JevBackend(model="jev-1.14.0", image_mode="experimental"),
)
```

Each image has a validated base64 `data_url`, a `detail` value of `"auto"`
(default), `"low"`, `"high"`, or `"original"`, and an optional `label`. Supported
MIME types in the shared schema are `image/png`, `image/jpeg`, `image/webp`, and
`image/gif`; each backend applies its own limits. Use
`ImageInput.from_file(path, detail="auto", label=None)` for local files or
`ImageInput.from_bytes(data, mime_type="image/png", detail="auto", label=None)`
for PNG bytes. `mime_type` is required; choose the type matching the supplied bytes.
The file helper infers it from the file suffix. Validation checks the supported
MIME type and nonempty base64 data; image decoding and format verification belong
to the receiving provider. HTTP image URLs and provider file IDs are not accepted; inline
bytes make sample evidence portable and stable for calibration hashing. The helper
reads a local file when creating the object, so later changes to that file do not
change the sample's image.

In JSON/JSONL, each image is an object with `data_url`, `detail`, and `label`:

```json
{"input":"Describe the receipt.","response":"The total is $42.00.","images":[{"data_url":"data:image/png;base64,<base64-image-bytes>","detail":"auto","label":"receipt"}]}
```

Replace the placeholder with actual base64 image bytes. The CLI's built-in metric
panels select text evidence; define image-selecting metrics in Python.

Single-judgment requests include the union of fields required by their active
metrics. Adding an image to a sample does not change built-in presets or text-only requests. When any active
metric selects images, single-judgment questions share those images in their
request. Context-relevance unit requests contain no image evidence.
An empty image tuple is missing evidence for an image-selecting metric and follows
the existing `missing="raise"` or `missing="skip"` policy.

`JevBackend` defaults to `image_mode="disabled"`. Opt in with `"experimental"`
to send the current experimental image header. Jev accepts PNG and JPEG images
with `detail="auto"`; unsupported formats and detail settings fail before the
batch opens a session. The adapter turns image evidence into SDK `Image` objects,
and the SDK sends multipart assets and inserts their references, preserving
image order and label mapping. TypeSafe documents image support for Jev 1.14,
the default model. See the
[TypeSafe image guide](https://docs.typesafe.ai/images),
[Jev 1.14 announcement](https://typesafe.ai/blog/jev-1-14), and
[runnable Jev example](../examples/jev_images.py).

When an endpoint supports the same image transport without an experimental flag,
use `JevBackend(model="...", image_mode="native")`. Native mode omits that header;
it does not discover model capabilities automatically. The mode is explicit and
does not depend on a model-version allowlist, so graduation of the existing image
API needs only configuration. A future change to the wire API or SDK contract
could still require an adapter update.

`OpenAIDecisionsBackend` advertises text and image support. It sends a user message
with JSON text evidence and native `input_image` parts, preserving image order,
detail, and label mapping. Decisions allows at most 128 selected images per sample;
that limit belongs to the adapter, not the shared sample schema.

For a compatible System One vision deployment, replace `your-vision-model` with
the model identifier accepted by your server:

```python
from typed_evals import SystemOneBackend

backend = SystemOneBackend(
    model="your-vision-model",
    base_url="http://localhost:30000",
    image_mode="native",
)
# Pass backend to evaluate(..., images=[...], metrics=[visual_grounding]).
```

`SystemOneBackend` defaults to `image_mode="disabled"`; native mode requires both
a vision-capable model and an endpoint implementing the
[SGLang image extension](https://docs.sglang.io/docs/supported-models/decision_models#image-inputs).
It sends ordered `{"url": data_url, "detail": detail}` objects in a top-level
`images` field, with indexed label/detail metadata in the JSON state. It accepts
the shared PNG, JPEG, WebP, and GIF inputs and `"auto"`, `"low"`, or `"high"`
detail; `"original"` fails before the session opens. The receiving server
determines byte, dimension, and image-processing limits. The adapter does not
resize or transform images and sends no Jev experimental feature header. This
extension is explicit, with no model allowlist or capability discovery; keep
images disabled when the server or model lacks support. See the
[runnable System One image example](../examples/systemone_images.py).

Selected images with an unsupported backend fail before its session opens;
unselected images remain usable in a text evaluation. Custom backends extending
`BaseBackend` use its shared `image_inputs(state)` parser, can override
`supported_modalities` to return
`frozenset({"text", "image"})` and translate the same evidence state without
changing the sample, metric, or evaluator APIs. Image transport settings belong
to the provider adapter; Jev supports disabled/experimental/native modes and
System One supports disabled/native modes.
Unsupported modalities and provider input-limit failures are preflight errors,
including with `errors="record"`.

Reports and calibration artifacts do not store raw image data. Image evidence,
order, detail, and labels enter overlap hashing when selected; use the same visual
rubric and backend configuration for fitting and inference. See
[image calibration compatibility](CALIBRATION.md#image-evidence-and-calibration).

## What the numbers mean

| Field | Meaning |
|---|---|
| `raw_score` | Noul probability of true; Choice probability mass of `pass_options`; normalized expected Score level; or context relevance's relevant-unit fraction |
| `raw_kind` | `event_probability`, `normalized_ordinal_score`, or `relevance_fraction`; identifies the raw signal's meaning |
| `confidence` | Provider-supplied Choice/Score confidence; absent for Noul/predicate. Its meaning depends on the model and serving runtime |
| `calibrated_probability` | Learned estimate of the probability that a human labels this **specific metric** as passing |
| `score` | Calibrated probability when available, otherwise raw score |
| `passed` | Whether `score >= metric.threshold`; `None` for skipped/error results |
| `details` | For context relevance, typed source-span verdicts, unit probabilities, counts, and coverage; no raw source text |

Calibration learns `raw_score → P(human metric label = 1)`. It does not train on
the provider's `confidence`, produce the generating LLM's confidence, or estimate whether
every decision made by the judge is correct. A normalized ordinal Score and a
relevant-unit fraction are not probabilities. Their calibrated outputs estimate
the probability of a binary human acceptance label.

With Decisions, predicate `0.90` produces raw score `0.90`. For Choice, passing
option probabilities `0.35 + 0.25` produce `0.60` even if another individual option
wins. Expected Score `1.5` over levels `0, 1, 2` produces raw score `0.75`; the
metric parser normalizes it exactly once. Confidence never multiplies these scores
and is not substituted for calibration input. Do not assume confidence has the
same semantics across providers.

See [backend translation](ARCHITECTURE.md#openai-decisions-api-translation) for the
provider wire format and answer validation contract.

For example, if examples with raw faithfulness around 0.9 have only 70% human
passes, a representative fitted curve may map that region toward 0.7. That number
is specific to the question, label rubric, model, and data distribution.

There is deliberately **no combined “probability the whole answer is true.”**
Per-metric probabilities are dependent; averaging them does not produce that
probability. `SampleResult.passed` is a policy conjunction of complete metric
results. It is `None` if any metric was skipped or has status `"error"`. If all
metrics are available and at least one misses its threshold, it is `False`.

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
The function is also available from `typed_evals.metrics`. There are eleven
built-in metrics:

| Metric | Evidence required | Human label 1 means |
|---|---|---|
| `AnswerRelevancy()` | `input`, `response` | The response directly addresses the request |
| `AnswerCompleteness()` | `input`, `response` | Every essential requested part is delivered; plans, promises, and refusals alone do not pass |
| `Faithfulness()` | `input`, `response`, `contexts` | Every material factual assertion is supported by the passages; input only resolves references and ellipsis |
| `AnswerCorrectness()` | `input`, `response`, `reference` | Essential facts match the reference answer |
| `ContextRelevance()` | `input`, `contexts` | The retrieved evidence has an acceptable relevant-unit fraction under the application's labeling rubric |
| `ContextSufficiency()` | `input`, `contexts` | The passages establish enough information to answer every essential part of the request |
| `TaskCompletion()` | `input`, `trace`, `expected_outcome` | Execution evidence establishes the intended outcome |
| `ToolGrounding()` | `response`, `trace` | Claims about tool results are supported by observed outputs |
| `ToolSafety(policy=...)` | `input`, `proposed_tool_call`, `contexts`, `authorization` | The proposed call complies with policy and application-owned authorization facts |
| `ToolAccuracy()` | `input`, `proposed_tool_call`, `contexts` | Tool selection and argument values match the supplied specifications and facts |
| `PolicyCompliance(policy=...)` | `input`, `response` | Candidate content complies with the application policy |

`ToolAccuracy()` uses `kind="choice"` with `"true"` and `"false"` options and
`pass_options=("true",)`. Its raw score is the probability assigned to `"true"`;
the default passing threshold is `0.8`. `ToolSafety` and `PolicyCompliance` default
to `0.9`; the other built-ins default to `0.5`. The response, RAG, and agent
presets use the constructors' defaults and do not add correctness or policy
checks implicitly.

These are explicitly defined judge rubrics with documented score semantics.
`Faithfulness()` returns a probability for the whole-response support event:
`0.9` does not mean 90% of claims are supported. `ContextRelevance()` returns
`relevant_units / total_units`, using deterministic source sentence units by
default. Its `0.9` means 90% of those units met `unit_threshold`, not a 90%
probability that the retrieval is acceptable. Calibrating either raw signal
estimates human acceptance of its documented metric. These definitions do not
claim numeric equivalence or greater empirical accuracy than Ragas or DeepEval.

A response without material factual claims may pass Faithfulness, and a response
without tool-result claims may pass ToolGrounding, including an empty response.
Evaluate response delivery alongside these grounding dimensions. A relevant
clarification or inability explanation can pass AnswerRelevancy while leaving
AnswerCompleteness unsatisfied. Context relevance measures judge-estimated
relevant content density;
it does not establish source truth, ranking quality, or reference-answer recall.
The RAG preset includes completeness and sufficiency, while the agent preset adds
relevance and completeness to its execution checks. Empty output therefore has
an explicit failing criterion even when execution succeeded. Completeness judges
coverage, not truth or proof that an action happened. Sufficiency judges evidence
coverage, not passage ranking or retrieval precision.
Reference correctness is relative to the reference, not an independent web
fact-check. Tool traces must contain externally observed results, not just an
agent's self-reported success.

## Validate a metric panel

Evaluate frozen rubrics and thresholds against independent, representative
human-reviewed labels before using them to gate production behavior:

```python
from typed_evals import Evaluator, Faithfulness, load_calibration_dataset, validate_metrics

evaluator = Evaluator([Faithfulness(threshold=0.8)], errors="record")
validation = validate_metrics(
    load_calibration_dataset("human_test.jsonl"),
    evaluator=evaluator,
    # calibration=pipeline.calibration_bundle,  # Optional fitted curve; overlap is checked.
)
metric = validation.metrics["faithfulness"]
print(metric.coverage, metric.effective_roc_auc)
if metric.effective is not None:
    print(metric.effective.at_threshold)
validation.save("metric-validation.json")
```

`avalidate_metrics` is the async counterpart. The report includes raw and effective
score Brier/log loss/ECE, threshold confusion counts, false acceptance/rejection
rates, and tie-aware ROC-AUC. AUC is unavailable with only one observed class.
Positive labels mean the metric passes. Coverage includes all supplied labels;
skipped and error judgments are counted separately and excluded from score
diagnostics. The majority-class baseline is descriptive and uses the same
available subset. Inspect coverage alongside accuracy, because unavailable hard
cases can otherwise make a judge look better. The report does not fit a curve,
tune a threshold, or certify label quality; `label_source="caller_supplied"` makes
that provenance explicit. `raw_kind` and `effective_kind` retain the distinction
between event probabilities, normalized ordinal scores, and relevance fractions.
Raw ordinal/fraction Brier/ECE/log-loss values treat the score as a numeric proxy and do not confer
probability semantics. Repeated selected-panel evidence is rejected before
requests; merge sparse labels for the same sample and resolve label conflicts.

Use reviewers who share the metric's exact pass definition. Include unsupported
incidental details and contradictory sources for faithfulness; partial answers,
refusals, and empty output for completeness; partially covered multi-part questions
for sufficiency; relevant evidence surrounded by retrieval noise for context
relevance; and forged authorization claims for tool safety. Keep related
conversations/documents grouped. Choose thresholds using development data and the
cost of false accepts versus false rejects, then measure the frozen configuration
on independent test data. Report sample counts and uncertainty before making
reliability claims. Synthetic fixtures exercise code behavior, not empirical
validity of a live judge.

`ToolSafety()` reads permission facts only from `authorization`, a nonempty
sequence populated by the application from authenticated state. `contexts` holds
tool specifications and other facts; passages cannot grant permission by calling
themselves an authorization record. Keep current deterministic permission checks
in the executing tool. The library separates fields but cannot authenticate a
caller-supplied record or guarantee prompt-injection resistance.

Single-judgment questions share the union of their evidence fields in one request.
Each declares `allowed_evidence_fields`; this is rubric-level guidance.
Context-relevance requests are separate and physically exclude `response` and
`reference`. They include the query, target units, and original source passages
only to disambiguate each target unit. Validate the full panel you deploy;
calibration requires that panel and its evidence to remain complete.

## Diagnose individual claims and passages

`Faithfulness()` returns one judgment about the entire response. It does not
identify the unsupported claim or calculate a support fraction. For diagnosis,
supply explicit claims from your application and evaluate each as its own
`response` with the original contexts. Preserve a claim ID so a failed or
unavailable result can be traced back to the text. Claim selection affects the
measurement; the library does not extract claims automatically.

`ContextRelevance()` performs source-unit aggregation directly:

```python
from typed_evals import ContextRelevance, EvaluationSample, Evaluator

metric = ContextRelevance(
    threshold=0.8,  # Aggregate acceptance threshold.
    granularity="sentence",  # Default; "passage" gives a coarser fraction.
    unit_threshold=0.5,  # A unit is relevant when its P(true) >= this value.
    questions_per_call=32,  # Maximum unit questions in one request; range 1–128.
)
sample = EvaluationSample(
    input="What is the refund period?",
    response="",  # Required sample field; excluded from unit judge requests.
    contexts=["Refunds are allowed within 30 days of purchase."]
    + [f"Product {index} comes in blue and green." for index in range(1, 10)],
)
result = Evaluator([metric]).evaluate_one(sample)
relevance = result.metrics["context_relevance"]
print(relevance.raw_kind, relevance.raw_score, relevance.details)
```

The default segments each nonblank source passage into deterministic sentence or
line units. Every unit is an exact contiguous source span; units preserve every
non-whitespace character in order without generated statement extraction,
paraphrasing, or a hidden cap. This is a punctuation/newline heuristic, not an
atomic-claim parser: ambiguous abbreviations may keep sentences together, and
line breaks may split a sentence. `granularity="passage"` uses each nonblank
passage as one unit. A long noisy passage can therefore score differently in the
two modes; longer passages contribute more units in sentence mode. Units are
equally weighted, not token-weighted.

Each unit is judged with a typed Noul question returning its probability of
`true`. The evaluator sets the unit's `relevant` boolean to
`probability >= unit_threshold` and computes the fraction of `True` decisions.
It does not average probabilities or request generated relevance explanations,
reasons, or rationales. With `unit_threshold=0.5`, probabilities `0.9` and `0.4`
produce decisions `True` and `False`, hence raw fraction `0.5`.

For ten one-sentence passages with exactly one judged relevant, `raw_score=0.1`
regardless of that passage's rank. Unit relevance uses the query; the answer and
reference are excluded from these requests. Original source passages accompany
units only to resolve references or ellipsis; another useful sentence cannot
make an unrelated target unit relevant. On-topic fabricated statements can still
be relevant: this metric does not verify their truth.

`MetricResult.details` is a `ContextRelevanceDetails` containing `granularity`,
`segmentation_version`, `unit_threshold`, total/evaluated/relevant/unavailable
counts, coverage, and every `ContextUnitResult`. `coverage` is
`evaluated_units / total_units`, or `None` when there are no units. Unit results
preserve zero-based
`passage_index`, per-passage `unit_index`, Python-character `start`/`end` offsets,
`probability`, boolean `relevant`, `status`, and a safe `error_type` when unavailable.
Unavailable units have no probability or relevance decision. Reports omit raw
source text; recover
the exact span from your retained `contexts[passage_index][start:end]`.

Explicit empty contexts produce raw fraction `0.0`, zero units, and no unit
requests. Blank source passages are rejected during sample validation.
Omitted/`None` contexts are unavailable evidence: the default raises, while
`missing="skip"` records a skip without a score. With `errors="record"`, any missing,
refused, or malformed unit judgment, or any failed unit request, makes the
**whole aggregate unavailable**: status `"error"`, no score, and `passed=None`.
Successful unit diagnostics and sibling metrics remain available; failed units
remain in the counts. The denominator never silently shrinks. Default
`errors="raise"` raises instead. With calibration, missing panel evidence always
raises before requests, even with `missing="skip"`.

The [granular RAG example](../examples/granular_rag_evaluation.py) demonstrates
direct relevance aggregation, both granularities, and separate explicit-claim
faithfulness diagnosis. Its offline verdicts are synthetic fixtures.

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
enter the judge request. Each label requires the corresponding metric's evidence.
Sparse labels are allowed, provided each configured metric meets the split minima.
Calibration requires evidence for every configured metric on every row, even
where that metric's label is absent.

Omitted or `null` `contexts` and `trace` mean missing evidence. Explicit empty
arrays mean a known empty retrieval result or a recorded run with no tool events;
these are evaluated rather than skipped. Thus empty retrieval can fail relevance
and sufficiency, and a known empty execution trace can fail task completion.
`authorization` and selected `images` still require nonempty evidence.

Use `group_id` to keep the same conversation, source document, customer case, or
near-duplicate family together. Exact duplicate sample content is rejected even
if IDs differ. Automatic splitting preserves groups; explicit train/validation
group overlap is rejected. Evaluation reusing fitting/validation content or groups
is rejected by default. `allow_calibration_overlap=True` exists for inspecting
already-seen rows; those results are not independent performance measurements.

## Custom metrics

```python
from typed_evals import Metric, evaluate

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

result = evaluate(
    input="What is the refund period, and how do I request one?",
    response="You have 30 days. Contact support with your order number.",
    metrics=[completeness],
)
```

Describe Score levels from least to most compliant: higher normalized scores must
mean better outcomes because passing always uses `score >= threshold`.
`pass_definition` does not reverse the score direction. For Choice, use a mapping
of descriptions and explicitly set `pass_options`. For Noul, ask one crisp positive criterion and
optionally supply `criteria={"true": "...", "false": "..."}`. See
[custom_metrics.py](../examples/custom_metrics.py). `pass_definition` documents the
binary target for annotators; the question and criteria are what the backend judges.
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
materialized before evaluation. The wrapper rejects generators, async generators,
and unresolved awaitables before calling `sample_builder` or the evaluator; callers
must consume streams or await results explicitly. `evaluated_by` evaluates completed outcomes.
Use the runtime APIs below to enforce checks inside the execution loop.

For multiple tool calls, let `sample_builder` return a list or tuple of
`EvaluationSample` objects, one per call. `evaluated.evaluation` then contains an
`EvaluationReport`; inspect its `.results` for individual scores and `.summary`
for aggregates. Returning one sample still produces a `SampleResult`. An empty
list produces an empty report with no judge requests.

<details>
<summary>Recipe: evaluate every recorded tool call in a sequential agent</summary>

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

</details>

## CLI and CI

```bash
typed_evals evaluate examples/assets/rag_samples.jsonl \
  --metrics faithfulness answer_relevancy --output report.json

typed_evals calibrate examples/assets/labeled.jsonl \
  --metrics faithfulness answer_relevancy --output calibration.json

typed_evals evaluate examples/assets/test.jsonl --metrics faithfulness answer_relevancy \
  --calibration calibration.json --threshold 0.8 --output report.json --fail-on-failure

# Uses gpt-6-luna when --model is omitted:
typed_evals evaluate examples/assets/rag_samples.jsonl --backend openai-decisions \
  --metrics faithfulness answer_relevancy --errors record --output openai-report.json

typed_evals calibrate examples/assets/labeled.jsonl --backend openai-decisions \
  --metrics faithfulness answer_relevancy --output openai-calibration.json
```

Exit codes: `0` completed; `1` CI gate failed (including skipped/error/empty
evaluations); `2` execution/configuration error. Enable the CI gate with
`--fail-on-failure`; without it, a completed report exits `0` even when checks fail.
The CLI supports built-ins that do not require a policy. Construct
`PolicyCompliance(policy=...)`, `ToolSafety(policy=...)`, and custom metrics in
Python. `--threshold` applies the same threshold to every selected metric when
provided. Omitting it preserves each constructor's default, including `0.8` for
`ToolAccuracy()`.
By default, missing evidence and API failures raise.
Use `missing="skip"` / `--missing skip` or `errors="record"` / `--errors record`
explicitly when partial reports are appropriate. Errors and skips are excluded
from means, with separate counts.

Both commands support `--backend jev` (default), `--backend openai-decisions`, and
`--backend systemone`. Jev and Decisions select their default when `--model` is
omitted; System One requires `--model` and `--base-url`, with optional
`--api-key-env`. Fit and load calibration with the same backend configuration.

To evaluate with Microsoft Decision-1 on OpenRouter, map your exported OpenRouter
key to the SDK's environment variable and provide the model ID:

```bash
TYPESAFE_BASE_URL='https://openrouter.ai/api' \
TYPESAFE_API_KEY="$OPENROUTER_API_KEY" \
typed_evals evaluate examples/assets/rag_samples.jsonl --backend jev \
  --model microsoft/microsoft-decision-1 --metrics answer_relevancy \
  --output openrouter-report.json
```

The CLI reads environment variables; it does not load `.env` itself.

A Decisions refusal raises by default. With `errors="record"` / `--errors record`,
the refused metric has status `"error"`, no score, and no pass decision, while
successful sibling metrics remain available. Any unavailable metric leaves
`SampleResult.passed` as `None`. Independently attributable malformed answers use
the same per-metric policy; ambiguous response structure fails the whole request.

Exceptions are available from `typed_evals.errors`. `DecisionRefusalError`
subclasses `InvalidAnswerError` and identifies an unavailable per-question refusal;
provider refusal text is withheld. SDK transport failures become
`OpenAIDecisionsError` after the SDK's configured retries. Its message includes
only the SDK exception type, with provider response bodies, credentials, and
exception chaining suppressed. Recorded reports export safe exception types.
Invalid response structures raise `InvalidAnswerError`; cancellation propagates
as cancellation. This distinction keeps transport failure separate from a refused
or malformed judgment without exposing echoed input or secrets.

System One transport failures raise `SystemOneError` after bounded retries,
including only the HTTP status or transport exception type. Malformed or missing
answers are unavailable per metric; unknown answer names and duplicate JSON
fields fail the request. With `errors="record"`, successful sibling answers survive
independently invalid answers. Usage keeps scalar counters and ignores provider
diagnostics such as Laya's truncation flags.

## Sources checked

- [Official TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [TypeSafe image guide](https://docs.typesafe.ai/images)
- [Jev 1.14 announcement](https://typesafe.ai/blog/jev-1-14)
- [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- [OpenAI Python Decisions create reference](https://developers.openai.com/api/reference/python/resources/decisions/methods/create)
- [OpenAI Decisions create reference](https://developers.openai.com/api/reference/resources/decisions/methods/create)
- [TypeSafe confidence semantics](https://docs.typesafe.ai/confidence)
- [TypeSafe Score primitive](https://docs.typesafe.ai/primitives/score)
- [SGLang decision image inputs](https://docs.sglang.io/docs/supported-models/decision_models#image-inputs)
- [Laya HTTP API](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md)
- [Strands Decider server and models](https://github.com/strands-labs/strands-decider)
- [Strands System One wire schema](https://github.com/strands-labs/strands-decider/blob/main/src/strands_decider/schema.py)
- [Microsoft Decision-1 deployment and API guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)
- [Anthus Jev calibration experiment](https://anth.us/blog/can-you-trust-jev-confidence/)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

The native Decisions contract was checked on **10 October 2026** against the
three official OpenAI references above and installed SDK `openai==3.27.0`;
the supported minimum is `3.26.0`. The Jev contract is checked against
`typesafe-sdk==0.7.5`. Tests establish code
behavior; real-world metric accuracy and calibration quality require your own
labeled data. Judges can misjudge long, ambiguous, or adversarial inputs. The supplied
evaluator instructions are not a proven prompt-injection defense. Model/context
token limits still apply; the framework does not silently truncate evidence.
System One serving contracts were checked against the linked official Laya
and Strands sources on **10 October 2026**. Their serving runtimes may
truncate inputs; configure their limits for the evidence you evaluate.
The SGLang System One image extension was checked on **17 October 2026**.
The Foundry Decision-1 endpoint and authentication were checked against Microsoft's
guide on **17 October 2026**; no live Foundry deployment was used for verification.
