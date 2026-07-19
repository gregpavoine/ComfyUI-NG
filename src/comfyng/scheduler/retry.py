from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.0
    max_delay_seconds: float = 60.0
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name in ("base_delay_seconds", "max_delay_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base_delay_seconds cannot exceed max_delay_seconds")
        if not isinstance(self.retryable_exceptions, tuple) or not all(
            isinstance(item, type) and issubclass(item, BaseException)
            for item in self.retryable_exceptions
        ):
            raise ValueError("retryable_exceptions must contain exception types")

    def should_retry(self, *, attempt: int, error: BaseException) -> bool:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError("attempt must be a non-negative integer")
        if not isinstance(error, BaseException):
            raise ValueError("error must be an exception")
        return attempt + 1 < self.max_attempts and isinstance(
            error, self.retryable_exceptions
        )

    def delay_for_retry(self, *, next_attempt: int) -> float:
        if (
            isinstance(next_attempt, bool)
            or not isinstance(next_attempt, int)
            or next_attempt < 1
        ):
            raise ValueError("next_attempt must be a positive integer")
        if self.base_delay_seconds == 0:
            return 0.0
        if next_attempt > 1024:
            return float(self.max_delay_seconds)
        try:
            delay = self.base_delay_seconds * 2 ** (next_attempt - 1)
        except OverflowError:
            return float(self.max_delay_seconds)
        return min(self.max_delay_seconds, delay)


__all__ = ["RetryPolicy"]
