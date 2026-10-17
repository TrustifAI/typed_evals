"""Evaluate with a deployed Jev-compatible System One server.

Start your chosen Clef, Laya, Strands, or compatible server, then run:
    python examples/systemone_decisions.py --model Cloudflare/clef-flash \
        --base-url http://localhost:30000
For a protected endpoint, also pass --api-key-env MY_SERVER_API_KEY.
Makes one live request. Does not download or start a model server.
"""

import argparse
import os

from typed_evals import SystemOneBackend, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model identifier accepted by the server")
    parser.add_argument("--base-url", required=True, help="API root before /v1/systemone")
    parser.add_argument(
        "--api-key-env", help="Environment variable containing an optional bearer key"
    )
    args = parser.parse_args()
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key or not api_key.strip():
            parser.error("The environment variable named by --api-key-env must contain an API key")

    backend = SystemOneBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        timeout=60.0,
    )
    result = evaluate(
        input="What is the refund period?",
        response="You can request a refund within 30 days.",
        contexts=["Refunds are allowed within 30 days of purchase."],
        preset="rag",
        backend=backend,
    )
    print("All checks passed:", result.passed)
    for name, metric in result.metrics.items():
        print(f"{name}: raw_score={metric.raw_score}, score={metric.score}, passed={metric.passed}")


if __name__ == "__main__":
    main()
