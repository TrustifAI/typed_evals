"""Evaluate image evidence with a deployed System One vision server.

Run from a checkout with a vision-capable model and an endpoint accepting
SGLang's top-level images extension, for example a compatible Clef deployment:
    python examples/systemone_images.py receipt.png \\
        --model Cloudflare/clef-flash --base-url http://localhost:30000 \\
        --input "What total is shown?" --response 'The total is $42.00.'

Makes one live request. Does not download or start a model server.
For a protected endpoint, also pass --api-key-env MY_SERVER_API_KEY.
"""

import argparse
import os
from pathlib import Path

from typed_evals import ImageInput, Metric, SystemOneBackend, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG, JPEG, WebP, or GIF file")
    parser.add_argument(
        "--model", required=True, help="Vision-capable model served by the endpoint"
    )
    parser.add_argument("--base-url", required=True, help="API root before /v1/systemone")
    parser.add_argument(
        "--api-key-env", help="Environment variable containing an optional bearer key"
    )
    parser.add_argument("--input", default="Describe the supplied image.")
    parser.add_argument("--response", required=True, help="Candidate response to evaluate")
    parser.add_argument("--detail", choices=("auto", "low", "high"), default="auto")
    args = parser.parse_args()

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key or not api_key.strip():
            parser.error("The environment variable named by --api-key-env must contain an API key")

    visual_grounding = Metric(
        name="visual_grounding",
        kind="noul",
        instructions=(
            "Does response answer input with visual claims supported by the supplied images?"
        ),
        pass_definition="Every material visual claim is supported by the supplied images.",
        required_fields=("input", "response", "images"),
        threshold=0.8,
    )
    result = evaluate(
        input=args.input,
        response=args.response,
        images=[ImageInput.from_file(args.image, detail=args.detail, label="source image")],
        metrics=[visual_grounding],
        backend=SystemOneBackend(
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            image_mode="native",
            timeout=60.0,
        ),
    )
    print("All checks passed:", result.passed)
    for name, metric in result.metrics.items():
        print(f"{name}: raw_score={metric.raw_score}, score={metric.score}, passed={metric.passed}")


if __name__ == "__main__":
    main()
