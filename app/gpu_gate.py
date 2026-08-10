from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.config import settings

T = TypeVar("T")

_gpu_semaphore: asyncio.Semaphore | None = None


def get_gpu_semaphore() -> asyncio.Semaphore:
    """Lazy-init so the semaphore binds to the running event loop."""
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(settings.max_concurrent)
    return _gpu_semaphore


async def run_on_gpu(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, int]:
    """
    Run sync GPU inference off the event loop, gated by a concurrency semaphore.

    Returns (result, queue_wait_ms).
    """
    sem = get_gpu_semaphore()
    queued_at = time.perf_counter()
    async with sem:
        queue_wait_ms = int((time.perf_counter() - queued_at) * 1000)
        result = await asyncio.to_thread(fn, *args, **kwargs)
        return result, queue_wait_ms
