from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from threading import Event, RLock


class CancellationCheckpoint(StrEnum):
    SAMPLER_STEP = "sampler_step"
    BETWEEN_BLOCKS = "between_blocks"
    BEFORE_DECODE = "before_decode"
    BEFORE_SAVE = "before_save"


@dataclass(frozen=True, slots=True)
class CheckpointObservation:
    checkpoint: CancellationCheckpoint
    position: int | None = None


class CancellationRequested(asyncio.CancelledError):
    def __init__(
        self,
        reason: str,
        checkpoint: CancellationCheckpoint,
        position: int | None,
    ) -> None:
        self.reason = reason
        self.checkpoint = checkpoint
        self.position = position
        super().__init__(f"cancelled at {checkpoint.value}: {reason}")


class CancellationToken:
    def __init__(self, *, history_limit: int = 128) -> None:
        if (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 1
        ):
            raise ValueError("history_limit must be a positive integer")
        self._event = Event()
        self._reason: str | None = None
        self._history: deque[CheckpointObservation] = deque(maxlen=history_limit)
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()
        self._lock = RLock()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def history(self) -> tuple[CheckpointObservation, ...]:
        with self._lock:
            return tuple(self._history)

    def cancel(self, reason: str = "cancelled") -> bool:
        if not isinstance(reason, str) or not reason or reason != reason.strip():
            raise ValueError("reason must be a non-empty trimmed string")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            waiters = tuple(self._waiters)
        for loop, waiter in waiters:
            try:
                loop.call_soon_threadsafe(waiter.set)
            except RuntimeError:
                # A waiter whose event loop has already closed cannot observe the
                # notification, but must not prevent cancellation for live workers.
                continue
        return True

    def checkpoint(
        self,
        checkpoint: CancellationCheckpoint | str,
        *,
        position: int | None = None,
    ) -> None:
        try:
            resolved = CancellationCheckpoint(checkpoint)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unknown cancellation checkpoint: {checkpoint!r}"
            ) from exc
        if position is not None and (
            isinstance(position, bool) or not isinstance(position, int) or position < 0
        ):
            raise ValueError("position must be a non-negative integer or None")
        with self._lock:
            self._history.append(CheckpointObservation(resolved, position))
            reason = self._reason
        if reason is not None:
            raise CancellationRequested(reason, resolved, position)

    async def wait(self) -> str:
        with self._lock:
            if self._reason is not None:
                return self._reason
            loop = asyncio.get_running_loop()
            waiter = asyncio.Event()
            key = (loop, waiter)
            self._waiters.add(key)
        try:
            await waiter.wait()
            with self._lock:
                return self._reason or "cancelled"
        finally:
            with self._lock:
                self._waiters.discard(key)


__all__ = [
    "CancellationCheckpoint",
    "CancellationRequested",
    "CancellationToken",
    "CheckpointObservation",
]
