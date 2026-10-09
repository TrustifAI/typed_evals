# Calibration

## Opt into automated calibration

Calibration is **off by default**. When enabled, the pipeline must be fitted or
loaded before evaluation; it never silently falls back to raw scores.

To try the example below, run it from the repository root with the supplied
[`labeled.jsonl`](../examples/assets/labeled.jsonl) and
[`test.jsonl`](../examples/assets/test.jsonl). These fictional
customer-support/RAG datasets contain 160 labeled rows across 40 scenario groups
and 20 test rows across five separate groups. Each labeled group covers all four
pass/fail combinations for `faithfulness` and `answer_relevancy`. Related responses
share a `group_id`, so the default split keeps them together and produces 128
training rows and 32 held-out rows, with both classes for each metric.

The supplied labels are **synthetic, assistant-authored demo expectations**, not
human annotations; provenance is recorded in each sample's `metadata`. Use them
to exercise the pipeline, and replace them with representative, human-reviewed
labels before interpreting calibration probabilities or quality improvements.
The test file contains evaluation samples without labels and shares no scenario
groups or sample content with the labeled file. Running this example uses the
Jev API and requires `TYPESAFE_API_KEY` and the calibration dependencies above.

```python
from typed_evals import (
    CalibrationConfig,
    EvaluationPipeline,
    Faithfulness,
    AnswerRelevancy,
    load_calibration_dataset,
    load_dataset,
)

pipeline = EvaluationPipeline(
    metrics=[Faithfulness(threshold=0.8), AnswerRelevancy(threshold=0.8)],
    calibration=CalibrationConfig(enabled=True),
    max_concurrency=8,
)

# Automatically: split labeled data, judge it, fit per-metric curves,
# measure them on held-out rows, then evaluate the independent test set.
report = pipeline.run(
    load_dataset("examples/assets/test.jsonl"),
    calibration_data=load_calibration_dataset("examples/assets/labeled.jsonl"),
)
pipeline.save_calibration("calibration.json")
report.save("evaluation-report.json")
print(pipeline.calibration_report)
```

The default automatic hold-out uses 20% of independent sample groups. Each metric
needs at least **100 labeled training rows**, **20 held-out rows**, and both
classes on each side. These are input safeguards, not a statistical guarantee.
Use several hundred representative expert-labeled examples when possible, and
inspect the held-out report. You can explicitly lower the minima for experiments.

For controlled or temporal splits, provide your own independent validation set:

```python
diagnostics = pipeline.fit(
    load_calibration_dataset("fit.jsonl"),
    validation_data=load_calibration_dataset("validation.jsonl"),
)
report = pipeline.evaluate(load_dataset("examples/assets/test.jsonl"))
```

The deployed curve uses **only the training partition**. The pipeline does not
refit on the hold-out after reporting performance. Reports include raw and
calibrated Brier score, log loss, binary ECE, reliability bins, and accuracy at
0.5. Improvement is measured, not assumed. A worse held-out Brier score is
explicitly recorded; fitting does not automatically approve a model for deployment.

Reload without training again:

```python
production = EvaluationPipeline(
    metrics=[Faithfulness(threshold=0.8), AnswerRelevancy(threshold=0.8)],
    calibration=CalibrationConfig(enabled=True),
).load_calibration("calibration.json")

result = production.evaluate_one(sample)  # A sample outside the fitting/validation data
```

Changing metric wording, criteria, evidence fields, version, question panel/order,
backend/compiler configuration, or the requested/observed model version invalidates
the artifact. Changing an
acceptance threshold does not. Save custom metric definitions in your own source
alongside the artifact. JSON knots are validated and restored without pickle.

## Fit Decisions calibration separately

Install `python -m pip install 'typed-evals[openai,calibration]'` and set
`OPENAI_API_KEY`. Pass the same Decisions backend for fitting and later inference:

```python
from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    EvaluationPipeline,
    Faithfulness,
    OpenAIDecisionsBackend,
    load_calibration_dataset,
)

pipeline = EvaluationPipeline(
    [Faithfulness(threshold=0.8), AnswerRelevancy(threshold=0.8)],
    backend=OpenAIDecisionsBackend(model="gpt-6-luna"),
    calibration=CalibrationConfig(enabled=True),
)
pipeline.fit(load_calibration_dataset("examples/assets/labeled.jsonl"))
pipeline.save_calibration("openai-calibration.json")

production = EvaluationPipeline(
    [Faithfulness(threshold=0.9), AnswerRelevancy(threshold=0.9)],
    backend=OpenAIDecisionsBackend(model="gpt-6-luna"),
    calibration=CalibrationConfig(enabled=True),
).load_calibration("openai-calibration.json")
```

Or use the CLI, which selects `gpt-6-luna` when `--model` is omitted:

```bash
typed_evals calibrate examples/assets/labeled.jsonl --backend openai-decisions \
  --metrics faithfulness answer_relevancy --output openai-calibration.json
typed_evals evaluate examples/assets/test.jsonl --backend openai-decisions \
  --metrics faithfulness answer_relevancy --calibration openai-calibration.json \
  --threshold 0.9 --output openai-report.json
```

This uses the same algorithm and raw-score target as Jev. Decisions Choice/Score
confidence remains a separate provider value and never enters fitting. A labeled
refusal or unavailable judgment fails the fit; the previous successful in-memory
bundle remains intact. Model identity and dataset/group overlap checks continue
to apply.

## Artifact versions and compatibility

New fits save `schema_version: 2` with explicit `backend_provenance`. Bundled
backends supply a verified configuration and its SHA-256 fingerprint: provider,
backend identity, compiler version, score-affecting static configuration, and the
ordered compiled questions. The fingerprint describes translation rules and
question configuration, not sample contents. Credentials, timeouts, retries, and
decision thresholds are excluded. Changing a threshold alone therefore remains
compatible; changing provider, compiler, or question translation requires a refit.
Decisions also binds calibration to its effective endpoint, including an injected
client's endpoint or `OPENAI_BASE_URL`; credentials in URL userinfo, query, and
fragment are excluded. Keep any additional caller-controlled routing behavior
stable when reusing a fitted curve.

Third-party backends implementing only `model` and `session()` remain supported.
Their new artifacts carry `status: "unknown"`, rather than fabricated provider
metadata, and retain the existing model, metric, ordering, evidence, and overlap
checks. Unknown provenance is compatible only with unknown provenance; it cannot
be transferred to a verified bundled provider. For verified v2 artifacts, direct
`CalibrationBundle.validate_for(...)` callers must also supply current backend
provenance. Evaluators and pipelines supply it automatically.

V1 artifacts remain readable without rewriting or inventing historical metadata.
They retain established compatibility with Jev's original compiler (`"1"`) and
custom backends that expose no provenance capability, subject to all existing
checks. They cannot prove Decisions compatibility, even if a model ID is edited
to match. Loading one with Decisions or another provenance-aware backend requires
a fresh fit from labeled data and gives a clear refit error. There is no bypass
switch or automatic conversion of an old curve into verified Decisions calibration.

## Image evidence and calibration

Image-selecting metrics use the same human pass/fail labels and fitting algorithm
as text metrics. Include `"images"` in their `required_fields`, put `ImageInput`
objects in each sample's `images`, and fit with an image-capable backend. Keep
representative image quality, detail settings, and visual tasks in both training
and held-out data. A curve fitted on a text rubric does not establish calibration
for a new visual rubric.

The reserved-sample hashes include the selected images' inline content, order,
detail, and optional labels. Changing only sample IDs or a local source file's
path does not make the same image evidence independent. These checks detect exact
evidence duplication, not perceptually similar images, crops, or re-encoded copies;
use `group_id` for related images and versions of the same source document.
Unselected images do not enter text-only evidence hashes.

Image-selecting metrics use metric fingerprint schema version 3, and Decisions
records its multimodal compiler configuration for those panels. Existing
text-only panels retain their established metric fingerprints and Decisions
compiler identity, so adding image support does not itself invalidate text-only
artifacts. Changing required fields or provider translation still requires a
refit. Compiler provenance describes how images are translated, not the image
data in any particular sample.

Saved calibration artifacts and evaluation reports contain no raw image payloads.
As with text evidence, selected images are supplied to the judge provider for
fitting and inference. Save the labeled dataset separately if you need to
reproduce a fit.

## The statistical target

For each metric m, annotate independent samples with y_m ∈ {0, 1}, where 1 means
the sample satisfies the metric's documented pass definition. Let s_m be its raw
judge signal. The fitted monotone mapping estimates:

`g_m(s_m) ≈ P(y_m = 1 | s_m)`.

This is event-probability calibration, not top-label judge-accuracy calibration.
For a Noul with raw value 0.1, the input is 0.1, not max(0.1, 0.9) and not 0.8
“decisiveness.” For Choice, the input is the summed probability of its explicitly
accepted options. For Score, the input is the expected level divided by the
maximum level. A Score's target must be defined as a binary human acceptance
event; it is not fitted against fractional ordinal labels.

## Algorithm

1. Validate metric label names, evidence, independent groups, and class/sample counts.
2. Reserve a hold-out or accept an explicitly supplied independent validation set.
3. Collect raw metric scores without any existing calibrator. Labels never enter state.
4. Fit `sklearn.isotonic.IsotonicRegression(increasing=True, y_min=0, y_max=1,
   out_of_bounds="clip")` per metric using only labeled training rows.
5. Evaluate the fitted curves on labeled validation rows. Publish raw/calibrated
   Brier score, log loss, ECE, and reliability bins for each metric.
6. Atomically replace the in-memory bundle after all metrics succeed. Save it
   explicitly to a versioned JSON artifact for reuse.
7. For unseen samples, check metric, model, and backend provenance, then interpolate the
   stored knots and apply each metric's configured threshold.

The stored predictor matches scikit-learn's linear interpolation between learned
knots, including clipping outside the observed fitting range. No refitting and
no scikit-learn import occurs during inference. Constant raw scores yield a
constant training base rate; they cannot gain discrimination through calibration.

## Data separation

The automatic split shuffles independent groups deterministically. With no
`group_id`, each unique sample is a group. It is **not stratified** for all possible
multi-metric sparse label combinations. The pipeline checks class counts after
splitting and rejects insufficient splits before API calls. For rare positives,
temporal drift, or unequal group sizes, provide an explicit suitable split.

Use `group_id` for related conversations, document versions, users, or templates
whenever their overlap would exaggerate generalization. The framework detects
exact duplication of the metric panel's evidence and explicitly declared groups;
changing IDs or fields unused by the metric panel does not bypass this check. It does not discover
semantic duplicates. Keep an additional test set independent if you tune prompts,
select metrics, or repeatedly tune thresholds on the validation report.

All configured metrics are fitted or the fit fails. Sparse labels are supported:
only rows labeled for a metric enter that metric's fit and diagnostics. Missing
evidence is allowed only for unlabeled metrics, which are skipped on that row.
Do not label every sample automatically as “pass”; both classes are required.

## Diagnostics

- **Brier**: mean squared difference between predicted probability and binary label.
- **Log loss**: binary cross entropy, with numerical clipping at 1e-15 and 1−1e-15.
- **ECE**: equal-width positive-event probability bins, weighted by count, averaging
  the absolute gap between mean probability and observed positive frequency.
- **Accuracy at 0.5**: pass/fail classification accuracy at a fixed probability
  cutoff. This is separate from each metric's production acceptance threshold.

ECE is bin-dependent and noisy on small samples; it is not a confidence interval.
The artifact includes bin counts and observed rates so empty/tiny bins are visible.
Isotonic fitting is monotone but can introduce ties, change threshold decisions,
and sometimes worsen held-out performance. It cannot repair a fundamentally
uninformative judge. Calibration on one domain is not a guarantee for another.

## Operational contract

The bundle records requested and observed model IDs, backend/compiler provenance,
ordered metric fingerprints,
fitting counts, validation metrics, creation time, and hashes of reserved samples
and declared groups. Fingerprints include wording, criteria, required fields,
pass options, pass definition, metric version, and the evaluation instruction
policy. Thresholds are excluded because they change routing, not the raw score
or the annotation event.

Refit after model/rubric changes or data-distribution changes. Model/configuration
changes are detected; distribution drift is not automatically detected. If the
provider changes weights behind the same version string, this library cannot
discover that through version checking alone. Keep historical held-out reports
and periodically evaluate a fresh labeled sample.

Artifacts are JSON, not executable pickle files. They must still come from a
trusted source: a structurally valid but maliciously edited curve can change
decisions. There is no cryptographic artifact signing or automated retraining
service in this package.
