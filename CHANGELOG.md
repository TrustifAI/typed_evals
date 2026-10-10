# Changelog

This file records changes from its introduction onward. Earlier release history
has not been reconstructed. See the
[repository tags](https://github.com/TrustifAI/typed_evals/tags) and
[package index](https://pypi.org/project/typed-evals/)
for published versions.

The published baseline checked for this update is
[0.2.0 on PyPI](https://pypi.org/project/typed-evals/0.2.0/).

## Unreleased

### Added

- `SystemOneBackend` connects to Jev-compatible `/v1/systemone` servers,
  including Clef, Laya, and Strands Decider, with optional bearer authentication
  and explicit model and endpoint configuration. Python and CLI support local
  servers without an API key; a runnable example documents setup.

### Changed

- Malformed choice values and oversized numeric answers now produce metric
  errors instead of crashing partial reports; probability-sum tolerance is
  capped for large choice rubrics.
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

- Documented the incompatible `openai` and `crewai` extras beside installation
  and adapter instructions, with separate-environment guidance.
- Added concise use cases and navigation to the evaluation and runtime references.
- Moved notebook import/reload troubleshooting to the [FAQ](docs/FAQ.md).
- Documented whole-response faithfulness and whole-context relevance limits, with
  a recipe for evaluating explicit claims and individual passages.
- Added contribution and security reporting guidance.
- Corrected links to the runnable OpenRouter example.

## Migration notes

### Package imports

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

### Runtime metadata

Use `metadata["typed_evals"]` for runtime audit data. The previous
`metadata["jev"]` key remains an alias to the same payload, including when a
Decisions or custom backend supplies the judgment. Model information belongs in
the evaluation result, not in the metadata container name. See the
[runtime reference](docs/RUNTIME.md#backend-selection-and-metadata).

### Changing judge backends

Jev remains the default. Select `OpenAIDecisionsBackend()` in Python or
`--backend openai-decisions` in the CLI for native Decisions; its SDK is in the
`openai` extra. Hosted TypeSafe-compatible models use `JevBackend` with explicit
model, endpoint, and credentials. Use `SystemOneBackend` or `--backend systemone`
for other deployed System One servers, including ones without authentication.
See [backend configuration](docs/EVALUATION.md#choose-a-backend).

Refit calibration when changing provider, model, rubric, or score-affecting
compiler configuration. Read the [artifact compatibility contract](docs/CALIBRATION.md#artifact-versions-and-compatibility)
before reusing a saved calibration bundle.
