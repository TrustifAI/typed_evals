"""Evaluate image evidence with OPENAI_API_KEY and typed-evals[openai].

Run from a checkout:
    python examples/openai_decisions_images.py receipt.png \
        --input "What total is shown?" --response 'The total is $42.00.'

Makes a paid Decisions API request. The image is read into a portable ImageInput;
the custom metric selects it alongside the text evidence.
"""

import argparse
from pathlib import Path

from typed_evals import ImageInput, Metric, OpenAIDecisionsBackend, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a response against a local image.")
    parser.add_argument("image", type=Path, help="PNG, JPEG, WebP, or GIF file")
    parser.add_argument("--input", default="Describe the supplied image.")
    parser.add_argument("--response", required=True, help="Candidate response to evaluate")
    parser.add_argument("--detail", choices=("auto", "low", "high", "original"), default="auto")
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
        images=[ImageInput.from_file(args.image, detail=args.detail, label="source image")],
        metrics=[visual_grounding],
        backend=OpenAIDecisionsBackend(model="gpt-6-luna", timeout=30.0, max_retries=2),
    )
    print("All checks passed:", result.passed)
    for name, metric in result.metrics.items():
        print(f"{name}: raw_score={metric.raw_score}, score={metric.score}, passed={metric.passed}")


if __name__ == "__main__":
    main()
