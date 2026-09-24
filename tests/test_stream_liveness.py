import asyncio

import pytest

from codex_shim.stream_liveness import with_liveness


@pytest.mark.asyncio
async def test_heartbeat_does_not_restart_upstream():
    release = asyncio.Event()
    starts = 0
    beats = 0

    async def source():
        nonlocal starts
        starts += 1
        await release.wait()
        yield {"type": "completed"}

    async def heartbeat():
        nonlocal beats
        beats += 1
        release.set()

    events = source()
    got = [event async for event in with_liveness(events, disconnected=lambda: False,
                                                heartbeat=heartbeat, interval=0.001)]
    await events.aclose()
    assert got == [{"type": "completed"}]
    assert starts == beats == 1


@pytest.mark.asyncio
async def test_disconnect_cancels_pending_model_work():
    closed = asyncio.Event()
    disconnected = False

    async def source():
        try:
            await asyncio.Event().wait()
            yield {}
        finally:
            closed.set()

    async def heartbeat():
        nonlocal disconnected
        disconnected = True

    events = source()
    with pytest.raises(ConnectionResetError):
        async for _ in with_liveness(events, disconnected=lambda: disconnected,
                                     heartbeat=heartbeat, interval=0.001):
            pass
    await events.aclose()
    assert closed.is_set()
