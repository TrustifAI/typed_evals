# FAQ and troubleshooting

For API contracts, use the [evaluation guide](EVALUATION.md),
[runtime guide](RUNTIME.md), and [adapter guide](ADAPTERS.md). See the
[changelog](../CHANGELOG.md#migration-notes) for import and metadata migration notes.

## Why are my contexts not being checked?

Without a preset or explicit metrics, evaluation checks answer relevancy using
only `input` and `response`. The `"response"` preset also checks completeness.
Supplying `contexts` does not change either panel. Choose `preset="rag"`
or pass metrics such as `Faithfulness()` and `ContextRelevance()` explicitly.
Evaluation emits one aggregate `UserWarning` per batch when active metrics do not
use supplied contexts; skipped metrics do not count as consuming evidence.
Only selected evidence enters judge requests. Context relevance derives unit
spans and source-passage context from the selected `contexts`; it excludes the
generated response and reference answer from its separate requests.
`passed=True` covers the selected checks only: a relevant answer can contradict
unused contexts. Use `preset="rag"` to include faithfulness before treating a
pass as evidence that the response agrees with those contexts.

## Why can a context relevance check pass with mostly bad passages?

`ContextRelevance()` now scores the fraction of source sentence units judged
relevant: one relevant unit among ten scores `0.1`. `unit_threshold` determines
each unit's boolean verdict from its typed true/false probability using
`probability >= unit_threshold`; the raw score counts those verdicts rather than
averaging probabilities. `threshold` determines whether the aggregate passes. Inspect
`MetricResult.details` for every source span, probability, and verdict. Optional
`granularity="passage"` is coarser: one useful sentence can make its whole passage
relevant. On-topic fabricated text can still be relevant because relevance does
not verify truth. Existing calibration panels containing this metric need a refit.

Sentence splitting uses deterministic punctuation and newline rules, not generated
claim extraction. A unit can contain several claims; passage mode is coarser
still. Diagnostics contain source offsets, not saved source text or generated
rationales. A missing, invalid, refused, or otherwise unavailable unit judgment
makes the entire metric unavailable. With `errors="record"`, its coverage and known unit verdicts remain
visible alongside successful sibling metrics; no partial fraction is reported.
Pair relevance with `ContextSufficiency()` to check whether the evidence covers
every essential part of the request.

`Faithfulness()` judges the whole response; evaluate explicit claims separately
when you need to locate unsupported text. See
[individual claims and passages](EVALUATION.md#diagnose-individual-claims-and-passages)
and the [granular RAG example](../examples/granular_rag_evaluation.py).

## Why are missing evidence and empty results treated differently?

Omitting `contexts` or `trace`, or supplying `None`, means the evidence is
unavailable. A metric requiring it raises by default or is skipped with
`missing="skip"`. An explicit empty sequence means retrieval or execution was
observed to return no results, so the metric evaluates that case. Empty contexts
give `ContextRelevance` a raw score of `0` without a judge request. An empty
response is also evaluated; grounding alone can pass when there are no claims,
while relevance and completeness check whether a response was delivered.

The tool decorators require nonempty `contexts` and a separate nonempty
`authorization` sequence. Permission facts belong in `authorization`; text in
retrieved passages or model arguments cannot grant permission. See the
[adapter evidence contract](ADAPTERS.md#shared-options).

## Does the agent preset check tool safety?

The `"agent"` preset checks observed task completion, tool-result grounding,
answer relevance, and answer completeness. Supply `input`, `response`, observed
`trace`, and an explicit `expected_outcome`. It does not include `ToolSafety` or
`ToolAccuracy`; select those explicitly at a before-tool checkpoint. The tool
decorators use `ToolSafety` with your application policy and permission evidence.
See [runtime metric panels](RUNTIME.md#metric-panels-and-evidence).

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

Use Python 3.12 or later, install the repository into the notebook kernel's
environment, and run the notebooks from the repository root. They use
`python-dotenv` to load `.env`; set credentials for the sections you run.

- The [introductory notebook](../notebooks/sample_notebook.ipynb) starts with Jev
  (`TYPESAFE_API_KEY`), then shows OpenRouter (`OPENROUTER_API_KEY`) and a local
  System One server.
- The [advanced notebook](../notebooks/advanced_usage.ipynb) demonstrates calibration;
  install the `calibration` extra to enable isotonic fitting. Its agent sections
  need the `agent-framework` and `langchain` extras, `langchain-google-genai`,
  and `GEMINI_API_KEY`; the LangChain
  section also accepts `GOOGLE_API_KEY`.

See the [contribution setup](../CONTRIBUTING.md#development-environment) for an
editable install. Add `python-dotenv`, a Jupyter kernel, and the packages needed
by your selected notebook to that same environment.

## Why did the package version stay the same after editing pyproject.toml?

`typed_evals.__version__` reads installed package metadata. After changing
`[project].version`, refresh an editable install with
`python -m pip install --no-deps -e .`, then restart any running notebook kernel.
See the [release instructions](../CONTRIBUTING.md#releases).

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
exact permissions and current state inside the tool. Supply authenticated
permission facts separately in `authorization`; field separation alone does not
authenticate them. A post-execution check can
withhold output but cannot undo effects. See
[dispatch behavior](RUNTIME.md#guard-tool-dispatch-before-execution) and
[failure semantics](RUNTIME.md#policy-and-failure-semantics).
