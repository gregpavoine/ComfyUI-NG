from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock


class PressureKind(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    PINNED_MEMORY = "pinned_memory"
    GPU = "gpu"
    IO_READ = "io_read"
    IO_WRITE = "io_write"


class PressureLevel(StrEnum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class PressureThresholds:
    elevated_ratio: float = 0.8
    critical_ratio: float = 0.95

    def __post_init__(self) -> None:
        for name in ("elevated_ratio", "critical_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
        if not 0 < self.elevated_ratio < self.critical_ratio <= 1:
            raise ValueError(
                "pressure ratios must satisfy 0 < elevated < critical <= 1"
            )


@dataclass(frozen=True, slots=True)
class PressureSample:
    kind: PressureKind
    used: int
    limit: int
    device_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PressureKind):
            raise ValueError("kind must be a PressureKind")
        if (
            isinstance(self.used, bool)
            or not isinstance(self.used, int)
            or self.used < 0
        ):
            raise ValueError("used must be a non-negative integer")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        if self.device_index is not None and (
            isinstance(self.device_index, bool)
            or not isinstance(self.device_index, int)
            or self.device_index < 0
        ):
            raise ValueError("device_index must be a non-negative integer or None")
        if self.kind is PressureKind.GPU and self.device_index is None:
            raise ValueError("GPU pressure requires device_index")
        if self.kind is not PressureKind.GPU and self.device_index is not None:
            raise ValueError("device_index is only valid for GPU pressure")

    @property
    def ratio(self) -> float:
        return self.used / self.limit


@dataclass(frozen=True, slots=True)
class PressureEvent:
    kind: PressureKind
    previous_level: PressureLevel
    level: PressureLevel
    used: int
    limit: int
    ratio: float
    device_index: int | None
    observed_at: datetime


class PressureMonitor:
    def __init__(self, thresholds: PressureThresholds | None = None) -> None:
        self.thresholds = thresholds or PressureThresholds()
        self._levels: dict[tuple[PressureKind, int | None], PressureLevel] = {}
        self._lock = RLock()

    def _classify(self, ratio: float) -> PressureLevel:
        if ratio >= self.thresholds.critical_ratio:
            return PressureLevel.CRITICAL
        if ratio >= self.thresholds.elevated_ratio:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    def observe(self, sample: PressureSample) -> PressureEvent | None:
        if not isinstance(sample, PressureSample):
            raise ValueError("sample must be a PressureSample")
        key = (sample.kind, sample.device_index)
        with self._lock:
            previous = self._levels.get(key, PressureLevel.NORMAL)
            level = self._classify(sample.ratio)
            self._levels[key] = level
        if level is previous:
            return None
        return PressureEvent(
            kind=sample.kind,
            previous_level=previous,
            level=level,
            used=sample.used,
            limit=sample.limit,
            ratio=sample.ratio,
            device_index=sample.device_index,
            observed_at=datetime.now(UTC),
        )

    def level(
        self,
        kind: PressureKind,
        *,
        device_index: int | None = None,
    ) -> PressureLevel:
        with self._lock:
            return self._levels.get((kind, device_index), PressureLevel.NORMAL)


__all__ = [
    "PressureEvent",
    "PressureKind",
    "PressureLevel",
    "PressureMonitor",
    "PressureSample",
    "PressureThresholds",
]
