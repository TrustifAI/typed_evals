"""Evaluate a response with Microsoft's Decision-1 model on OpenRouter.

Install python-dotenv and set OPENROUTER_API_KEY in your environment or .env.
Run from a checkout: python examples/openrouter_decisions.py
Makes one live System One API request through JevBackend.
"""

import os

from dotenv import load_dotenv

from typed_evals import JevBackend, evaluate


def main():
    load_dotenv()
    backend = JevBackend(
        model="microsoft/microsoft-decision-1",
        base_url="https://openrouter.ai/api",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    result = evaluate(
        input="What is the refund period?",
        response="You can request a refund within 30 days.",
        preset="response",
        backend=backend,
    )
    print("All checks passed:", result.passed)
    for name, metric in result.metrics.items():
        print(f"{name}: raw_score={metric.raw_score}, score={metric.score}, passed={metric.passed}")


if __name__ == "__main__":
    main()
