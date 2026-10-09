"""Optional evidence capabilities shared by all judge backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typed_evals.errors import UnsupportedModalityError


def validate_evidence(backend: object, state: Mapping[str, Any]) -> None:
    """Keep existing model/session-only backends text-only unless they opt in.

    Image-capable backends advertise ``supported_modalities = {"text", "image"}``
    and translate the neutral ``images`` evidence in their own adapter.
    """
    supported = getattr(backend, "supported_modalities", frozenset({"text"}))
    if state.get("images") and "image" not in supported:
        raise UnsupportedModalityError(
            f"{type(backend).__name__} does not support image evidence; "
            "use a backend that advertises the image modality"
        )
    validate = getattr(backend, "validate_state", None)
    if validate is not None:
        if not callable(validate):
            raise TypeError("Backend validate_state must be callable")
        validate(state)
