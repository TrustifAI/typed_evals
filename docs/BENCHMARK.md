# Typed Evals · TRIVIA+

**Jev before and after calibration**  
Live benchmark · 27 September 2026 · 645 test answers

> **A 68% smaller gap between confidence and results.**  
> Similar hallucination detection, with scores that better reflected human judgments.

---

## 1. The dataset

[TRIVIA+](https://github.com/amazon-science/hallucination-benchmark-trivialplus) contains **3,224 examples**: a source article, a question, an AI-generated answer, and human judgments about whether the article supports the answer.

An answer fails this check if any sentence contradicts the article or makes an unsupported claim. The test set contains **421 supported answers and 224 answers with hallucinations**, drawn from **358 articles**.

**Cleanup before calibration**

Some identical article–answer pairs appeared more than once. For training and validation, we kept one copy when human labels agreed and removed all copies when labels conflicted.

| Group | Purpose | Original rows | Rows used | Rows excluded |
|---|---|---:|---:|---:|
| Training | Learn the score adjustment | 2,263 | **2,195** | 68 |
| Validation | Check the adjustment and choose cutoffs | 316 | **308** | 8 |
| Test | Measure final performance | 645 | **645** | **0** |

The exclusions were **54 duplicate copies + 14 rows with conflicting labels** in training, and **6 duplicate copies + 2 rows with conflicting labels** in validation.

We preserved the dataset's original groups; no identical article appeared across groups. **All 645 test answers were retained**, including repeated answers and conflicting labels.

## 2. How the benchmark worked

1. **Judge the answer.** Typed Evals' Faithfulness check gave Jev the full article and answer. The question and human label were not part of the judge's input.
2. **Learn the adjustment.** Training examples taught Typed Evals how to adjust Jev's scores to better match human judgments.
3. **Choose the cutoffs.** Separate cutoffs for raw and calibrated scores were chosen using validation data, aiming for the best detection score (F1). Both were saved before testing.
4. **Test on the same answers.** Both versions used the same saved Jev judgments on all 645 test answers. Calibration required no additional judge calls.
5. **Check uncertainty.** Repeated samples of whole articles helped estimate how much the results could vary, keeping related answers together.

Calibration gives scores a more useful meaning. If answers receive an 80% chance of passing, roughly 80% should pass human review. Training data teaches the adjustment; test data measures the result.

## 3. The results

**How closely did the scores match human judgments?**

Lower values are better for all three checks below.

| Check | Raw Jev | Calibrated Jev | Reduction |
|---|---:|---:|---:|
| Gap between confidence and outcomes, averaged across score groups (ECE) | 0.0982 | **0.0313** | **68.1%** |
| Probability error, with extra weight on larger mistakes (Brier score) | 0.2037 | **0.1948** | **4.4%** |
| Penalty for confidently wrong predictions (log loss) | 0.6286 | **0.5765** | **8.3%** |

The confidence gap fell from **9.82 to 3.13 percentage points**. This is an improvement in the scores' reliability, not a 68% increase in detection accuracy.

The reported 95% uncertainty ranges supported all three improvements. These checks held the learned adjustment fixed and measured variation within this test set.

**How well did it detect hallucinations?**

A hallucination score above the cutoff triggers a flag. **F1** balances how many hallucinations are caught with how often those flags are correct. Higher is better.

| Jev scores | Cutoff choice | Hallucination cutoff | Detection score (F1) | Answers classified correctly |
|---|---|---:|---:|---:|
| Raw | Default | 0.5000 | 0.5258 | 71.47% |
| Calibrated | Default | 0.5000 | 0.4478 | 71.32% |
| Raw | Chosen on validation | 0.3900 | 0.5833 | **72.09%** |
| Calibrated | Chosen on validation | 0.2953 | **0.5877** | 71.94% |

At the default cutoff, calibration flagged fewer answers and missed more hallucinations. With cutoffs chosen on validation data, detection performance was very similar. The small F1 increase does not establish a meaningful advantage.

The ability to rank hallucinated answers above supported ones also showed no improvement. The measurable gain was in how well the probabilities matched outcomes.

## 4. Verdict

**Jev's scores became more reliable, while detection accuracy stayed around 72% after choosing suitable cutoffs.**

Typed Evals makes the adjustment reusable: learn it from representative human labels, save it, and apply it to future scores from the same judge and check. In this run, that meant better probability estimates without another judge call for each answer.

This supports the value of calibration on TRIVIA+. It does not establish a better hallucination detector or superiority over other judges, and results on other data may differ.

---

**Sources & scope**  
Dataset and cleanup: the pinned [TRIVIA+ release](https://github.com/amazon-science/hallucination-benchmark-trivialplus/tree/b71110b612f7ea7b2e65ff7af45b5338cece2fe5) and dataset audit. Results: the supplied live `RESULTS.md` and validation-selected cutoff table. Implementation: [Typed Evals](https://github.com/TrustifAI/typed_evals). Speed and cost were not measured.
