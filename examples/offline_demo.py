"""Runnable synthetic pipeline demonstration. No API calls and no accuracy claims."""

from contextlib import asynccontextmanager

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    CalibrationExample,
    EvaluationPipeline,
    EvaluationSample,
    JudgeResponse,
)


class SyntheticBackend:
    model = "synthetic-demo-not-jev"

    @asynccontextmanager
    async def session(self):
        yield self

    async def judge(self, state, questions):
        # A deterministic stand-in for a miscalibrated judge. Never use this in production.
        probability = float(state["response"])
        return JudgeResponse(
            model=self.model,
            answers={name: {"type": "noul", "noul": probability} for name in questions},
        )


def labeled_rows(prefix, size):
    rows = []
    for index in range(size):
        bucket = index % 2
        label = int((index // 2) % 5 < (2 if bucket == 0 else 3))
        rows.append(
            CalibrationExample(
                sample=EvaluationSample(
                    input=f"{prefix}-{index}", response=str(0.2 if bucket == 0 else 0.8)
                ),
                labels={"answer_relevancy": label},
            )
        )
    return rows


def main():
    pipeline = EvaluationPipeline(
        [AnswerRelevancy(threshold=0.7)],
        backend=SyntheticBackend(),
        calibration=CalibrationConfig(enabled=True),
    )
    report = pipeline.run(
        [EvaluationSample(input="independent-production-example", response="0.8")],
        calibration_data=labeled_rows("fit", 200),
        validation_data=labeled_rows("validation", 100),
    )
    metric = report.results[0].metrics["answer_relevancy"]
    diagnostics = pipeline.calibration_report.metrics["answer_relevancy"]
    print("Synthetic demonstration — these are not live Jev results")
    print(f"Raw score: {metric.raw_score:.2f}")
    print(f"Calibrated probability: {metric.calibrated_probability:.2f}")
    print(f"Pass at 0.7: {metric.passed}")
    print(f"Held-out Brier: {diagnostics.raw.brier:.3f} → {diagnostics.calibrated.brier:.3f}")
    print(f"Held-out ECE: {diagnostics.raw.ece:.3f} → {diagnostics.calibrated.ece:.3f}")


if __name__ == "__main__":
    main()
