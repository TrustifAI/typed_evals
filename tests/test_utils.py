import asyncio
import threading
from contextvars import ContextVar

import pytest

from typed_evals._utils import run_sync


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
