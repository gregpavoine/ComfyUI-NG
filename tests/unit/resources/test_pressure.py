from __future__ import annotations

import pytest

from comfyng.resources.pressure import (
    PressureKind,
    PressureLevel,
    PressureMonitor,
    PressureSample,
    PressureThresholds,
)


def test_pressure_monitor_emits_only_level_transitions() -> None:
    monitor = PressureMonitor(
        PressureThresholds(elevated_ratio=0.75, critical_ratio=0.9)
    )

    assert (
        monitor.observe(PressureSample(PressureKind.MEMORY, used=50, limit=100)) is None
    )
    elevated = monitor.observe(PressureSample(PressureKind.MEMORY, used=80, limit=100))
    assert elevated is not None
    assert elevated.previous_level is PressureLevel.NORMAL
    assert elevated.level is PressureLevel.ELEVATED
    assert elevated.ratio == 0.8
    assert (
        monitor.observe(PressureSample(PressureKind.MEMORY, used=85, limit=100)) is None
    )

    critical = monitor.observe(PressureSample(PressureKind.MEMORY, used=95, limit=100))
    assert critical is not None
    assert critical.previous_level is PressureLevel.ELEVATED
    assert critical.level is PressureLevel.CRITICAL

    recovered = monitor.observe(PressureSample(PressureKind.MEMORY, used=20, limit=100))
    assert recovered is not None
    assert recovered.previous_level is PressureLevel.CRITICAL
    assert recovered.level is PressureLevel.NORMAL


def test_pressure_is_tracked_independently_per_kind_and_device() -> None:
    monitor = PressureMonitor()
    gpu0 = PressureSample(PressureKind.GPU, used=95, limit=100, device_index=0)
    gpu1 = PressureSample(PressureKind.GPU, used=95, limit=100, device_index=1)

    first = monitor.observe(gpu0)
    second = monitor.observe(gpu1)

    assert first is not None and first.device_index == 0
    assert second is not None and second.device_index == 1
    assert monitor.level(PressureKind.GPU, device_index=0) is PressureLevel.CRITICAL
    assert monitor.level(PressureKind.GPU, device_index=1) is PressureLevel.CRITICAL


def test_invalid_pressure_thresholds_and_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        PressureThresholds(elevated_ratio=0.95, critical_ratio=0.9)
    with pytest.raises(ValueError):
        PressureSample(PressureKind.CPU, used=-1, limit=100)
    with pytest.raises(ValueError):
        PressureSample(PressureKind.CPU, used=1, limit=0)
