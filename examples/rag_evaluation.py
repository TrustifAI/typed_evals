"""Run with TYPESAFE_API_KEY set. Makes one API request per dataset sample."""

from pathlib import Path

from typed_evals import AnswerCorrectness, AnswerRelevancy, Faithfulness, evaluate


def main():
    # Supply custom metrics instead of a preset when reference correctness is needed.
    report = evaluate(
        Path(__file__).parent / "assets" / "rag_samples.jsonl",
        metrics=[Faithfulness(), AnswerRelevancy(), AnswerCorrectness()],
    )
    for result in report.results:
        print(result.sample_id, {name: metric.score for name, metric in result.metrics.items()})
    report.save("rag-report.json")


if __name__ == "__main__":
    main()
