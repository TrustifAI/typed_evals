"""Optional, static calibration identity without extending the Backend protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from typed_evals._utils import digest
from typed_evals.errors import CalibrationError

if TYPE_CHECKING:
    from typed_evals.backends.jev import Backend, Question


def backend_provenance(backend: Backend, questions: Mapping[str, Question]) -> dict[str, Any]:
    """Snapshot an optional compiler identity, never inference from a model name.

    Custom backends need only ``model`` and ``session()``. A backend may additionally
    provide ``calibration_provenance(questions)`` returning JSON-safe static settings.
    Its provider, backend, compiler version, and ordered question definitions identify
    the score-producing configuration; evidence, credentials, and transport settings
    must not be included.
    """
    describe = getattr(backend, "calibration_provenance", None)
    if describe is None:
        return {"status": "unknown"}
    if not callable(describe):
        raise CalibrationError("Backend calibration_provenance must be callable")
    configuration = describe(questions)
    try:
        validate_configuration(configuration)
        # Own a JSON snapshot: caller-owned nested configuration cannot mutate an
        # already fitted bundle or become a corrupt artifact after a failed refit.
        configuration = json.loads(json.dumps(configuration, ensure_ascii=False, allow_nan=False))
        fingerprint = digest(configuration)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("Backend returned invalid static calibration provenance") from exc
    return {"status": "verified", "configuration": configuration, "fingerprint": fingerprint}


def validate_configuration(configuration: Any) -> None:
    if not isinstance(configuration, dict):
        raise ValueError("Backend provenance configuration must be an object")
    if any(
        not isinstance(configuration.get(key), str) or not configuration[key].strip()
        for key in ("provider", "backend", "compiler_version")
    ):
        raise ValueError("Backend provenance needs provider, backend, and compiler_version")


def validate_provenance(value: dict[str, Any]) -> None:
    """Reject corrupt v2 metadata instead of trusting its stored fingerprint."""
    if value == {"status": "unknown"}:
        return
    if not isinstance(value, dict) or set(value) != {
        "status",
        "configuration",
        "fingerprint",
    }:
        raise ValueError("Invalid backend calibration provenance")
    if value["status"] != "verified":
        raise ValueError("Invalid backend calibration provenance status")
    validate_configuration(value["configuration"])
    if value["fingerprint"] != digest(value["configuration"]):
        raise ValueError("Backend calibration provenance fingerprint does not match")
