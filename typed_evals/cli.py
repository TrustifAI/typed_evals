from __future__ import annotations

import argparse
import json
import os
import sys

from typed_evals.backends import Backend, JevBackend, OpenAIDecisionsBackend, SystemOneBackend
from typed_evals.calibration import CalibrationConfig
from typed_evals.data.datasets import load_calibration_dataset, load_dataset
from typed_evals.evaluation.pipeline import EvaluationPipeline
from typed_evals.metrics import BUILTIN_METRICS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="typed_evals", description="Evaluate LLM/agent responses with a typed judge backend"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "calibrate"):
        sub = commands.add_parser(command)
        sub.add_argument("dataset")
        sub.add_argument(
            "--metrics", nargs="+", choices=sorted(BUILTIN_METRICS), default=["answer_relevancy"]
        )
        sub.add_argument(
            "--backend", choices=["jev", "openai-decisions", "systemone"], default="jev"
        )
        sub.add_argument(
            "--model",
            help="Model ID; required for systemone, otherwise defaults to the backend's model",
        )
        sub.add_argument(
            "--base-url", help="System One server URL; required for --backend systemone"
        )
        sub.add_argument(
            "--api-key-env", help="Environment variable containing a System One server API key"
        )
        sub.add_argument("--concurrency", type=int, default=8)
        sub.add_argument("--threshold", type=float, default=0.5)
        sub.add_argument("--output", required=True)
        if command == "evaluate":
            sub.add_argument("--calibration", help="A fitted calibration JSON artifact")
            sub.add_argument("--missing", choices=["raise", "skip"], default="raise")
            sub.add_argument("--errors", choices=["raise", "record"], default="raise")
            sub.add_argument(
                "--fail-on-failure", action="store_true", help="Exit 1 on failed/skipped/error rows"
            )
        else:
            sub.add_argument("--validation-data")
            sub.add_argument(
                "--algorithm", choices=["auto", "venn_abers", "isotonic"], default="auto"
            )
            sub.add_argument("--isotonic-min-samples", type=int, default=2000)
            sub.add_argument("--validation-fraction", type=float, default=0.2)
            sub.add_argument("--min-samples", type=int, default=100)
            sub.add_argument("--min-validation-samples", type=int, default=20)
            sub.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    api_key = None
    if args.backend == "systemone":
        if args.model is None or not args.model.strip():
            parser.error("--backend systemone requires --model")
        if args.base_url is None or not args.base_url.strip():
            parser.error("--backend systemone requires --base-url")
        if args.api_key_env is not None:
            api_key = os.environ.get(args.api_key_env)
            if api_key is None or not api_key.strip():
                parser.error("--api-key-env must name a nonempty environment variable")
    elif args.base_url is not None or args.api_key_env is not None:
        parser.error("--base-url and --api-key-env require --backend systemone")
    try:
        metrics = [BUILTIN_METRICS[name](threshold=args.threshold) for name in args.metrics]
        backend: Backend
        if args.backend == "systemone":
            assert args.model is not None and args.base_url is not None
            backend = SystemOneBackend(model=args.model, base_url=args.base_url, api_key=api_key)
        else:
            backend_class = JevBackend if args.backend == "jev" else OpenAIDecisionsBackend
            backend = backend_class(**({"model": args.model} if args.model is not None else {}))
        if args.command == "calibrate":
            pipeline = EvaluationPipeline(
                metrics,
                backend=backend,
                max_concurrency=args.concurrency,
                calibration=CalibrationConfig(
                    enabled=True,
                    algorithm=args.algorithm,
                    isotonic_min_samples=args.isotonic_min_samples,
                    min_samples=args.min_samples,
                    min_validation_samples=args.min_validation_samples,
                    validation_fraction=args.validation_fraction,
                    random_state=args.seed,
                ),
            )
            calibration_report = pipeline.fit(
                load_calibration_dataset(args.dataset),
                validation_data=load_calibration_dataset(args.validation_data)
                if args.validation_data
                else None,
            )
            pipeline.save_calibration(args.output)
            print(calibration_report.model_dump_json(indent=2))
        else:
            pipeline = EvaluationPipeline(
                metrics,
                backend=backend,
                max_concurrency=args.concurrency,
                calibration=CalibrationConfig(enabled=bool(args.calibration)),
                missing=args.missing,
                errors=args.errors,
            )
            if args.calibration:
                pipeline.load_calibration(args.calibration)
            report = pipeline.evaluate(load_dataset(args.dataset))
            report.save(args.output)
            print(
                json.dumps(
                    {name: value.model_dump() for name, value in report.summary.items()}, indent=2
                )
            )
            if args.fail_on_failure and (
                not report.results or any(row.passed is not True for row in report.results)
            ):
                return 1
    except Exception as exc:
        # Do not print provider response bodies (which may contain user data or credentials).
        print(
            f"typed_evals failed ({type(exc).__name__}). Check inputs, configuration, and API access; "
            "use the Python API for exception details.",
            file=sys.stderr,
        )
        if args.backend == "openai-decisions" and isinstance(exc, ImportError):
            print(
                "Install native Decisions support with: python -m pip install 'typed-evals[openai]'",
                file=sys.stderr,
            )
        return 2
    return 0
