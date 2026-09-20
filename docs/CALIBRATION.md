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
or the requested/observed model version invalidates the artifact. Changing an
acceptance threshold does not. Save custom metric definitions in your own source
alongside the artifact. JSON knots are validated and restored without pickle.

## The statistical target

For each metric m, annotate independent samples with y_m ∈ {0, 1}, where 1 means
the sample satisfies the metric's documented pass definition. Let s_m be its raw
Jev signal. The fitted monotone mapping estimates:

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
7. For unseen samples, check metric and model identity, then interpolate the
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

The bundle records requested and observed model IDs, ordered metric fingerprints,
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
