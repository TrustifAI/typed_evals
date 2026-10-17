# Changelog

This file records changes from its introduction onward. Earlier release history
has not been reconstructed. See the
[repository tags](https://github.com/TrustifAI/typed_evals/tags) and
[package index](https://pypi.org/project/typed-evals/)
for published versions.

## 0.2.2

### Added

- `AnswerCompleteness` and `ContextSufficiency` measure delivered response
  requirements and evidence coverage. Response, RAG, and agent presets include
  completeness; RAG also checks sufficiency, and agent checks response relevance.
- `validate_metrics` / `avalidate_metrics` compare frozen panels with independent
  caller-supplied labels, reporting coverage, threshold decision errors, ROC-AUC,
  and raw/effective score diagnostics. Calibration reports include confusion
  counts at the configured threshold and a constant training-prior Brier baseline.

- `SystemOneBackend` supports image evidence with `image_mode="native"` on
  vision-capable models served by endpoints accepting SGLang's top-level `images`
  extension. Images remain disabled by default for text-only deployments.
  The adapter preserves image order and labels, validates supported detail
  settings before requests, and records the image compiler and mode for
  calibration. A runnable `systemone_images.py` example demonstrates use.
- `BaseBackend.image_inputs(state)` provides shared image parsing and validation
  for Jev, OpenAI Decisions, System One, and custom adapters; provider image
  capabilities and wire formats remain in their respective backends.
- Public `BaseBackend` provides the common extension API for Jev, OpenAI
  Decisions, System One, and new provider adapters, with shared configuration
  validation and text-only capability defaults. Backend and judgment contracts
  live in `typed_evals.backends.base`; existing protocol implementations and Jev
  contract imports remain supported. The offline example demonstrates inheritance,
  and the Decision-1 example explicitly disables Jev image mode.
- `JevBackend` can opt into PNG/JPEG image evidence with
  `image_mode="experimental"`; `image_mode="native"` uses the same SDK asset
  transport without the experimental header for endpoints that support it.
  Image calibration records the compiler and mode. Existing sample, metric,
  and evaluator APIs are unchanged; images remain disabled by default.
- `SystemOneBackend` connects to Jev-compatible `/v1/systemone` servers,
  including Clef, Laya, and Strands Decider, with optional bearer authentication
  and explicit model and endpoint configuration. Python and CLI support local
  servers without an API key; a runnable example documents setup.

### Changed

- `ContextRelevance` now measures the fraction of deterministic source sentence
  units judged relevant; one relevant unit among ten scores `0.1`. Optional
  `granularity="passage"` measures a coarser passage fraction. `unit_threshold`
  controls each unit decision; `threshold` remains the aggregate decision policy.
  Separate requests exclude the response/reference and contain at most
  `questions_per_call` units (default 32). Existing worker concurrency bounds all
  requests. Each verdict is a boolean derived from the unit's Noul probability;
  no textual rationale is generated. Typed details retain exact source-span
  addresses and every verdict, probability, and coverage count without raw source
  text. Any unavailable unit makes the aggregate unavailable; its denominator never
  silently shrinks.
- Calibration fitting and inference require complete evidence for every configured
  metric on every row, including sparsely labeled rows. `missing="skip"` cannot
  change the active panel while a saved curve is applied.
- Faithfulness consistently judges material assertions and specifies unresolved
  contradiction handling. It uses input to resolve response references or ellipsis,
  with support restricted to contexts. Each question declares its allowed evidence fields.
- `ToolSafety` and guarded tool decorators now require separate application-owned
  `authorization` evidence; permission facts are no longer inferred from `contexts`.
- Omitted/`None` contexts and traces are missing evidence; explicit empty sequences
  are known empty retrieval/execution results and remain evaluable.
- CLI thresholds preserve each metric's default unless `--threshold` is supplied.

- The default Jev model is now `jev-1.14.0`; examples, documentation, and test
  fixtures use the same model version. Images remain disabled by default.
- `typed_evals.__version__` now reads installed distribution metadata; update
  only `[project].version` in `pyproject.toml`, with no separate package edit.
- Python 3.12 or later is now required, and the minimum `typesafe-sdk` version
  is 0.7.5. CI and development guidance use the supported Python versions.
- Malformed choice values and oversized numeric answers now produce metric
  errors instead of crashing partial reports; probability-sum tolerance is
  capped for large choice rubrics.
- Jev calibration provenance now identifies the effective API root, including
  caller-owned client routing. Older Jev artifacts without endpoint provenance
  require refitting; custom backends without provenance retain legacy compatibility.
- Dataset validation errors suppress rejected row values in ordinary formatted
  tracebacks while retaining row and field locations.
- `evaluated_by` rejects returned generators, async generators, and unresolved
  awaitables before mapping or judging output. Callers must materialize them first.
- Opt-in calibration now defaults to Venn–Abers below 2,000 labeled training
  rows per metric and isotonic at or above that count. Python and CLI options
  allow an explicit algorithm and configurable cutoff.
- New calibration artifacts use schema v3 and record the algorithm and held-out
  log-loss improvement; legacy isotonic artifacts remain readable.
- ToolSafety now explicitly distinguishes application authorization from
  retrieved content, tool descriptions, and user claims. An adversarial corpus
  covers evidence separation and dispatch enforcement, with opt-in live tests.
- Runtime audit metadata now has a provider-neutral `"typed_evals"` key; the
  legacy `"jev"` key remains an alias to the same payload.
- Evaluation emits a `UserWarning` when supplied contexts are unused by the
  selected metric panel, while retaining explicit preset selection.
- Added a mypy check to the development workflow and CI.

### Documentation

- Aligned CLI metric and threshold guidance, image validation descriptions,
  calibration requirements, and runtime metadata semantics with the implementation.
- Updated setup guidance for the introductory and advanced notebooks.
- Added Microsoft Foundry Decision-1 setup through `SystemOneBackend`, covering
  deployment endpoints, Azure API keys, and Entra bearer tokens.
- Documented the incompatible `openai` and `crewai` extras beside installation
  and adapter instructions, with separate-environment guidance.
- Added concise use cases and navigation to the evaluation and runtime references.
- Moved notebook import/reload troubleshooting to the [FAQ](docs/FAQ.md).
- Documented whole-response faithfulness and source-unit relevance semantics,
  with an offline example of claim diagnosis and direct relevance aggregation.
- Updated notebook preset descriptions, source-unit inspection, and separate
  authorization evidence; cleared outputs that used superseded metric definitions.
- Added contribution and security reporting guidance.
- Corrected links to the runnable OpenRouter example.

### Migration notes

#### Context relevance aggregation

Keep the existing `ContextRelevance` name and `context_relevance` result key. Its
raw score is now a `relevance_fraction`, not the probability that any passage is
useful. The default uses deterministic sentence/line units; choose
`granularity="passage"` for whole passages. Reports expose `ContextRelevanceDetails`
with source offsets and unit verdicts; retain source passages separately to
recover text. Empty retrieval scores zero without a unit request; omitted/`None`
contexts remain unavailable evidence.

Refit calibration panels containing `ContextRelevance`, including `preset="rag"`.
Identity includes aggregation protocol, segmentation version, granularity,
`unit_threshold`, and `questions_per_call`; changing these requires a refit.
The aggregate `threshold` alone remains compatible. Adding this machinery
preserves unrelated single-judgment metric identities. A calibrated relevance
score estimates human acceptance of retrieval, while `raw_score` retains its
fraction meaning and the individual unit probabilities remain uncalibrated.

#### Revised metrics and evidence

Refit calibration after this update: the revised evaluation policy, allowed
evidence fields, and changed rubrics invalidate prior metric identities. Changing
only a threshold remains compatible, but requires measuring the new decision
policy on independent labeled data.

`preset="response"` now selects relevance and completeness. Omitting both `metrics`
and `preset` keeps the single relevance check. RAG and agent presets are broader;
choose an explicit metric list when the narrower checks are appropriate.

Move authenticated permission facts from `contexts` into `authorization` on
`EvaluationSample` and every `guard_tool` helper. Keep tool specifications and
known facts in `contexts`, and deterministic permission checks inside the tool.
Use `None` or omit `contexts`/`trace` for unavailable evidence; use `[]` only when
retrieval or execution was observed to produce no passages/events.

#### Package imports

Prefer public imports from `typed_evals` or its subpackages. If your application
imports old implementation modules directly, use the current paths:

| Previous implementation path | Current path |
| --- | --- |
| `typed_evals.models` | `typed_evals.data.models` |
| `typed_evals.backend` | `typed_evals.backends` |
| `typed_evals.evaluator` | `typed_evals.evaluation.evaluator` |

The root package exports the public API, including `EvaluationSample`, `Metric`,
and `Evaluator`. Restart a notebook kernel after updating code; see the
[reload FAQ](docs/FAQ.md#notebook-imports-after-updating-package-code).

#### Runtime metadata

Use `metadata["typed_evals"]` for runtime audit data. The previous
`metadata["jev"]` key remains an alias to the same payload, including when a
Decisions or custom backend supplies the judgment. Model information belongs in
the evaluation result, not in the metadata container name. See the
[runtime reference](docs/RUNTIME.md#backend-selection-and-metadata).

#### Changing judge backends

Jev remains the default. Select `OpenAIDecisionsBackend()` in Python or
`--backend openai-decisions` in the CLI for native Decisions; its SDK is in the
`openai` extra. Hosted TypeSafe-compatible models use `JevBackend` with explicit
model, endpoint, and credentials. Use `SystemOneBackend` or `--backend systemone`
for other deployed System One servers, including ones without authentication.
See [backend configuration](docs/EVALUATION.md#choose-a-backend).

Refit calibration when changing provider, model, rubric, or score-affecting
compiler configuration. Read the [artifact compatibility contract](docs/CALIBRATION.md#artifact-versions-and-compatibility)
before reusing a saved calibration bundle.
