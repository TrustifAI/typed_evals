"""Evaluate PNG/JPEG image evidence with TYPESAFE_API_KEY.

Run from a checkout:
    python examples/jev_images.py receipt.png \\
        --input "What total is shown?" --response 'The total is $42.00.'

Makes a paid Jev API request. Use --image-mode native only when the endpoint
supports the same image transport without the experimental feature header.
"""

import argparse
from pathlib import Path

from typed_evals import ImageInput, JevBackend, Metric, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a response against a local image.")
    parser.add_argument("image", type=Path, help="PNG or JPEG file")
    parser.add_argument("--input", default="Describe the supplied image.")
    parser.add_argument("--response", required=True, help="Candidate response to evaluate")
    parser.add_argument("--model", default="jev-1.14.0")
    parser.add_argument("--image-mode", choices=("experimental", "native"), default="experimental")
    args = parser.parse_args()

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
        images=[ImageInput.from_file(args.image, label="source image")],
        metrics=[visual_grounding],
        backend=JevBackend(model=args.model, image_mode=args.image_mode),
    )
    print("All checks passed:", result.passed)
    for name, metric in result.metrics.items():
        print(f"{name}: raw_score={metric.raw_score}, score={metric.score}, passed={metric.passed}")


if __name__ == "__main__":
    main()
