"""Run with OPENAI_API_KEY set and typed-evals[openai] installed."""

from typed_evals import OpenAIDecisionsBackend, evaluate


def main():
    result = evaluate(
        input="What is the refund period?",
        response="You can request a refund within 30 days.",
        contexts=["Refunds are allowed within 30 days of purchase."],
        preset="rag",
        backend=OpenAIDecisionsBackend(model="gpt-6-luna", timeout=30.0, max_retries=2),
    )
    print("All checks passed:", result.passed)
    for name, metric in result.metrics.items():
        print(f"{name}: raw_score={metric.raw_score}, score={metric.score}, passed={metric.passed}")


if __name__ == "__main__":
    main()
