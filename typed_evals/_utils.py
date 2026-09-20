from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_sync(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run to completion, using a worker thread when the caller already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    # A notebook's loop cannot be nested. Create and close a separate loop in a
    # worker, preserving caller context and propagating the original exception.
    context = copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, lambda: asyncio.run(factory())).result()


def write_json(path: str | Path, value: Any) -> None:
    """Atomic local write; never serializes arbitrary Python objects or NaN."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
