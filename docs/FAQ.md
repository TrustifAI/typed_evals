# FAQ and troubleshooting

For API contracts, use the [evaluation guide](EVALUATION.md),
[runtime guide](RUNTIME.md), and [adapter guide](ADAPTERS.md). See the
[changelog](../CHANGELOG.md#migration-notes) for import and metadata migration notes.

## Why are my contexts not being checked?

The default `"response"` preset checks answer relevancy using only `input` and
`response`. Supplying `contexts` does not change that panel. Choose `preset="rag"`
or pass metrics such as `Faithfulness()` and `ContextRelevance()` explicitly.
Evaluation emits one aggregate `UserWarning` per batch when active metrics do not
use supplied contexts; skipped metrics do not count as consuming evidence.
Only fields required by active metrics enter the judge request.
`passed=True` covers the selected checks only: a relevant answer can contradict
unused contexts. Use `preset="rag"` to include faithfulness before treating a
pass as evidence that the response agrees with those contexts.

## Why can a context relevance check pass with mostly bad passages?

`ContextRelevance()` asks whether at least one passage is useful. To inspect each
passage, evaluate one sample per passage with the same question. Similarly,
`Faithfulness()` judges the whole response; evaluate explicit claims separately
when you need to locate unsupported text. See
[individual claims and passages](EVALUATION.md#diagnose-individual-claims-and-passages)
and the [granular RAG example](../examples/granular_rag_evaluation.py).

## Notebook imports after updating package code

If a nonempty list of `ToolSafety(...)` / `ToolAccuracy(...)` raises
`ValueError: metrics must contain at least one Metric`, restart the notebook kernel
and rerun the imports and guard construction. Older evaluator versions use a class
identity check that can fail after partial module reloads or loading the package
under multiple import names. Use the root-level package consistently:

```python
from typed_evals import Evaluator, GuardPolicy, RuntimeGuard, ToolAccuracy, ToolSafety
```

The current evaluator revalidates base metric instances from another reload or
import alias of the same source file. Invalid entries identify their list index
and type. Restarting is still needed to load updated code already cached in a
running kernel, and to refresh custom metric subclasses after reloading modules.

## What do the repository notebooks need?

Install the repository into the notebook kernel's environment and run the
notebooks from the repository root. They use `python-dotenv` to load `.env` and
make live requests with `TYPESAFE_API_KEY` by default. The introductory notebook
also includes Microsoft Decision-1 through OpenRouter with `OPENROUTER_API_KEY`.
Set only the credentials for the examples you intend to run.

Calibration sections need the `calibration` extra. The advanced agent sections
also need the `agent-framework` and `langchain` extras, `langchain-google-genai`,
and `GEMINI_API_KEY`; the LangChain section also accepts `GOOGLE_API_KEY`.
See the [contribution setup](../CONTRIBUTING.md#development-environment)
for an editable install; add `python-dotenv` and any notebook-specific packages
to that same environment.

## Which API should I call in a notebook or async server?

Use `await aevaluate(...)` or an evaluator's `aevaluate` / `aevaluate_one` methods
to keep the calling event loop responsive. Synchronous methods also work in a
running loop, but block the caller while a worker thread performs evaluation.
If you supply an async SDK client, evaluate asynchronously in that client's owning
event loop. See [client and concurrency behavior](EVALUATION.md#evaluate-a-response).

## Why did attaching an image not change the evaluation?

Built-in presets select text evidence. A custom metric must include `"images"`
in `required_fields`, and the backend must support image evidence. Selected images
with an unsupported backend fail before requests begin. See the
[image guide](EVALUATION.md#image-evidence).

## Why do CrewAI and native OpenAI Decisions need separate environments?

The supported CrewAI dependency requires OpenAI SDK `<3`, while the Decisions
backend requires `>=3.26.0,<4`. Install the CrewAI extra in a separate environment
until those dependency bounds are compatible. The
[contribution guide](../CONTRIBUTING.md#development-environment) shows the test setups.

## Does a passing guard replace authorization in the tool?

No. A judge checks the proposed call against the evidence you supplied. Enforce
exact permissions and current state inside the tool. A post-execution check can
withhold output but cannot undo effects. See
[dispatch behavior](RUNTIME.md#guard-tool-dispatch-before-execution) and
[failure semantics](RUNTIME.md#policy-and-failure-semantics).
