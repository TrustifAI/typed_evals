import asyncio
import json
import threading
from contextvars import ContextVar
from pathlib import Path

import pytest

from typed_evals._utils import run_sync, write_json


async def test_run_sync_preserves_context_and_closes_worker_loop():
    request_id = ContextVar("request_id", default="unset")
    token = request_id.set("notebook-request")
    caller_loop = asyncio.get_running_loop()
    caller_thread = threading.current_thread()

    async def work():
        await asyncio.sleep(0)
        value = request_id.get()
        request_id.set("worker-only")
        return value, asyncio.get_running_loop(), threading.current_thread()

    try:
        value, worker_loop, worker_thread = run_sync(work)
        assert value == "notebook-request"
        assert request_id.get() == "notebook-request"
        assert worker_loop is not caller_loop
        assert worker_loop.is_closed()
        assert worker_thread is not caller_thread
        assert not worker_thread.is_alive()
    finally:
        request_id.reset(token)


@pytest.mark.parametrize("error", [ValueError("failed"), asyncio.CancelledError()])
async def test_run_sync_propagates_errors_and_cancellation(error):
    worker_loops = []

    async def work():
        worker_loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        raise error

    with pytest.raises(type(error)) as caught:
        run_sync(work)
    assert caught.value is error
    assert worker_loops[0].is_closed()


def test_failed_atomic_report_replacement_preserves_existing_file_and_removes_temp(
    tmp_path, monkeypatch
):
    path = tmp_path / "reports" / "report.json"
    original = {"version": "previous", "response": "Paris"}
    write_json(path, original)
    previous_bytes = path.read_bytes()
    replacement = {"version": "next", "response": "東京"}
    attempted = []

    def fail_replace(source, destination):
        temporary = Path(source)
        attempted.append(temporary)
        assert temporary.parent == path.parent
        assert json.loads(temporary.read_text(encoding="utf-8")) == replacement
        assert Path(destination) == path
        raise OSError("disk refused replacement")

    with monkeypatch.context() as patch:
        patch.setattr("typed_evals._utils.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk refused replacement"):
            write_json(path, replacement)
    assert len(attempted) == 1
    assert not attempted[0].exists()
    assert list(path.parent.iterdir()) == [path]
    assert path.read_bytes() == previous_bytes

    write_json(path, replacement)
    assert json.loads(path.read_text(encoding="utf-8")) == replacement
