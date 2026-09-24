"""Keep buffered model streams alive without retrying paid work."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any


async def with_liveness(
    events: AsyncIterator[dict[str, Any]],
    *,
    disconnected: Callable[[], bool],
    heartbeat: Callable[[], Awaitable[None]] | None = None,
    interval: float = 5.0,
) -> AsyncIterator[dict[str, Any]]:
    """Poll connection liveness without cancelling/restarting an upstream read.

    The caller owns/ closes ``events``. The outstanding read is cancelled and
    awaited before exit, allowing subprocess-backed iterators to clean up.
    A heartbeat is an SSE comment, never a model token or a completion signal.
    """
    pending: asyncio.Task | None = None
    try:
        while True:
            if disconnected():
                raise ConnectionResetError("Client disconnected")
            if pending is None:
                pending = asyncio.create_task(anext(events))
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                if disconnected():
                    raise ConnectionResetError("Client disconnected")
                if heartbeat is not None:
                    await heartbeat()
                continue
            task, pending = pending, None
            try:
                event = task.result()
            except StopAsyncIteration:
                return
            yield event
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
