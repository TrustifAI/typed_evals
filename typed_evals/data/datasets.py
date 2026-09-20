"""JSON/JSONL interchange without pandas, and strict row validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from typed_evals.data.models import CalibrationExample, EvaluationSample


def _rows(path: str | Path) -> Iterator[tuple[int, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield line_number, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {path.name} at line {line_number}"
                        ) from exc
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON datasets must contain an array of rows")
        yield from enumerate(data, 1)
    else:
        raise ValueError("Dataset must have a .json or .jsonl extension")


def load_dataset(path: str | Path) -> list[EvaluationSample]:
    rows = []
    for index, data in _rows(path):
        try:
            rows.append(EvaluationSample.model_validate(data))
        except ValidationError as exc:
            # Field locations only: validation errors can contain the user's raw text.
            fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
            raise ValueError(f"Invalid evaluation row {index}, fields: {fields}") from exc
    return rows


def load_calibration_dataset(path: str | Path) -> list[CalibrationExample]:
    rows = []
    for index, data in _rows(path):
        try:
            rows.append(CalibrationExample.model_validate(data))
        except ValidationError as exc:
            fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
            raise ValueError(f"Invalid calibration row {index}, fields: {fields}") from exc
    return rows
