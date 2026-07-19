from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class PriorityFactors:
    user_priority: int
    queue_age_bonus: float = 0.0
    warm_model_bonus: float = 0.0
    cache_reuse_bonus: float = 0.0
    memory_pressure_penalty: float = 0.0
    estimated_duration_penalty: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.user_priority, bool)
            or not isinstance(self.user_priority, int)
            or not 0 <= self.user_priority <= 100
        ):
            raise ValueError("user_priority must be an integer between 0 and 100")
        for name in (
            "queue_age_bonus",
            "warm_model_bonus",
            "cache_reuse_bonus",
            "memory_pressure_penalty",
            "estimated_duration_penalty",
        ):
            value = _finite(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def priority_score(factors: PriorityFactors) -> float:
    if not isinstance(factors, PriorityFactors):
        raise ValueError("factors must be PriorityFactors")
    return (
        factors.user_priority
        + factors.queue_age_bonus
        + factors.warm_model_bonus
        + factors.cache_reuse_bonus
        - factors.memory_pressure_penalty
        - factors.estimated_duration_penalty
    )


def queue_age_bonus(
    *,
    enqueued_at: float,
    now: float,
    points_per_second: float = 1.0,
) -> float:
    enqueued = _finite("enqueued_at", enqueued_at)
    current = _finite("now", now)
    rate = _finite("points_per_second", points_per_second)
    if rate < 0:
        raise ValueError("points_per_second must be non-negative")
    return max(0.0, current - enqueued) * rate


__all__ = ["PriorityFactors", "priority_score", "queue_age_bonus"]
