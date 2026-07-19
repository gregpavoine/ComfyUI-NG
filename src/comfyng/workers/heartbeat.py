from __future__ import annotations

import math
import time
from collections.abc import Callable


class HeartbeatWatchdog:
    """Monotonic liveness tracker independent of child wall-clock values."""

    def __init__(
        self,
        timeout: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("heartbeat timeout must be positive and finite")
        self.timeout = timeout
        self._clock = clock
        self._last_seen = clock()

    @property
    def last_seen(self) -> float:
        return self._last_seen

    @property
    def age(self) -> float:
        return max(0.0, self._clock() - self._last_seen)

    def observe(self) -> float:
        self._last_seen = self._clock()
        return self._last_seen

    def expired(self) -> bool:
        return self.age > self.timeout
