"""Fit from real human labels, then evaluate a separate dataset.

Usage: python examples/calibrated_pipeline.py fit.jsonl validation.jsonl examples/assets/test.jsonl
Both metrics need enough labeled examples of both classes in each partition.
"""

import argparse

from typed_evals import (
    AnswerRelevancy,
    CalibrationConfig,
    EvaluationPipeline,
    Faithfulness,
    load_calibration_dataset,
    load_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit")
    parser.add_argument("validation")
    parser.add_argument("test")
    args = parser.parse_args()
    pipeline = EvaluationPipeline(
        [Faithfulness(threshold=0.8), AnswerRelevancy(threshold=0.8)],
        calibration=CalibrationConfig(enabled=True),
    )
    report = pipeline.run(
        load_dataset(args.test),
        calibration_data=load_calibration_dataset(args.fit),
        validation_data=load_calibration_dataset(args.validation),
    )
    pipeline.save_calibration("calibration.json")
    report.save("evaluation-report.json")
    print(pipeline.calibration_report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
