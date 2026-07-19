from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from comfyng.resources.broker import (
    AdmissionOutcome,
    FallbackAction,
    ResourceBroker,
    ResourceViolationCode,
)
from comfyng.resources.budgets import (
    ResourceAlternative,
    ResourceEstimate,
    ResourceLimits,
)
from comfyng.resources.hardware import (
    CpuInventory,
    GpuDevice,
    HardwareInventory,
    MemoryInventory,
)
from comfyng.resources.pressure import PressureKind, PressureLevel


GIB = 1024**3


def inventory(*, ram_gib: int = 32, free_vram_mb: int = 12_288) -> HardwareInventory:
    return HardwareInventory(
        cpu=CpuInventory(
            physical_cores=16,
            logical_cores=32,
            architecture="x86_64",
        ),
        memory=MemoryInventory(
            total_bytes=ram_gib * GIB,
            available_bytes=ram_gib * GIB,
            swap_total_bytes=8 * GIB,
            swap_free_bytes=8 * GIB,
        ),
        gpus=(
            GpuDevice(
                index=0,
                name="RTX fixture",
                total_vram_mb=12_288,
                free_vram_mb=free_vram_mb,
            ),
        ),
    )


def broker(**overrides: object) -> ResourceBroker:
    values: dict[str, object] = {
        "inventory": inventory(),
        "reserve_cpu_cores": 2,
        "reserve_ram_bytes": 4 * GIB,
        "reserve_vram_mb": 768,
        "max_parallel_heavy_jobs": 1,
    }
    values.update(overrides)
    return ResourceBroker(**values)  # type: ignore[arg-type]


def estimate(
    *,
    cpu_cores: int = 4,
    ram_gib: int = 2,
    vram_mb: int = 4_000,
    heavy_gpu: bool | None = None,
    alternatives: tuple[ResourceAlternative, ...] = (),
) -> ResourceEstimate:
    return ResourceEstimate(
        cpu_cores=cpu_cores,
        ram_bytes=ram_gib * GIB,
        vram_mb=vram_mb,
        gpu_index=0 if vram_mb else None,
        heavy_gpu=(vram_mb > 0 if heavy_gpu is None else heavy_gpu),
        alternatives=alternatives,
    )


def test_limits_preserve_reserved_system_cpu_ram_and_vram() -> None:
    instance = broker()

    assert instance.limits.cpu_cores == 14
    assert instance.limits.ram_bytes == 28 * GIB
    assert instance.limits.vram_mb_by_gpu == ((0, 11_520),)

    too_many_cores = instance.admit(estimate(cpu_cores=15, vram_mb=0))
    too_much_ram = instance.admit(estimate(cpu_cores=1, ram_gib=29, vram_mb=0))
    too_much_vram = instance.admit(estimate(cpu_cores=1, vram_mb=11_521))

    assert too_many_cores.outcome is AdmissionOutcome.REJECTED
    assert too_many_cores.violations == ("cpu_cores",)
    assert too_much_ram.outcome is AdmissionOutcome.REJECTED
    assert too_much_ram.violations == ("ram_bytes",)
    assert too_much_vram.outcome is AdmissionOutcome.REJECTED
    assert too_much_vram.violations == ("gpu[0].vram_mb",)


def test_limits_keep_one_compute_core_on_a_small_host() -> None:
    tiny = HardwareInventory(
        cpu=CpuInventory(physical_cores=2, logical_cores=2, architecture="arm64"),
        memory=MemoryInventory(
            total_bytes=8 * GIB,
            available_bytes=8 * GIB,
            swap_total_bytes=0,
            swap_free_bytes=0,
        ),
    )

    limits = ResourceLimits.from_inventory(
        tiny,
        reserve_cpu_cores=2,
        reserve_ram_bytes=4 * GIB,
        reserve_vram_mb=768,
    )

    assert limits.cpu_cores == 1


def test_only_one_heavy_gpu_job_is_admitted_per_device() -> None:
    instance = broker()
    first = instance.admit(estimate(vram_mb=2_000))
    second = instance.admit(estimate(vram_mb=2_000))

    assert first.admitted
    assert first.reservation is not None
    assert second.outcome is AdmissionOutcome.DEFERRED
    assert second.action is FallbackAction.SEQUENCE
    assert second.violations == ("gpu[0].heavy_jobs",)

    first.reservation.release()
    third = instance.admit(estimate(vram_mb=2_000))
    assert third.admitted


def test_reservation_release_is_idempotent_and_restores_capacity() -> None:
    instance = broker()
    decision = instance.admit(estimate(cpu_cores=14, ram_gib=28, vram_mb=11_520))
    assert decision.reservation is not None
    assert instance.available.cpu_cores == 0
    assert instance.available.ram_bytes == 0
    assert instance.available.vram_mb_by_gpu == ((0, 0),)

    assert decision.reservation.release() is True
    assert decision.reservation.release() is False
    assert instance.available == instance.limits


def test_reservation_context_manager_releases_on_exception() -> None:
    instance = broker()
    decision = instance.admit(estimate())
    assert decision.reservation is not None

    with pytest.raises(RuntimeError, match="fixture"):
        with decision.reservation:
            raise RuntimeError("fixture")

    assert decision.reservation.released
    assert instance.available == instance.limits


def test_concurrent_admission_cannot_overbook_resources() -> None:
    instance = broker()

    with ThreadPoolExecutor(max_workers=8) as executor:
        decisions = tuple(executor.map(lambda _: instance.admit(estimate()), range(8)))

    admitted = tuple(item for item in decisions if item.admitted)
    assert len(admitted) == 1
    assert sum(item.outcome is AdmissionOutcome.DEFERRED for item in decisions) == 7


def test_broker_emits_pressure_and_recovery_events() -> None:
    instance = broker()
    decision = instance.admit(estimate(cpu_cores=14, ram_gib=28, vram_mb=11_520))
    assert decision.reservation is not None

    pressure = instance.drain_pressure_events()
    by_kind = {(item.kind, item.device_index): item for item in pressure}
    assert by_kind[(PressureKind.CPU, None)].level is PressureLevel.CRITICAL
    assert by_kind[(PressureKind.MEMORY, None)].level is PressureLevel.CRITICAL
    assert by_kind[(PressureKind.GPU, 0)].level is PressureLevel.CRITICAL

    decision.reservation.release()
    recovery = instance.drain_pressure_events()
    assert {(item.kind, item.level) for item in recovery} >= {
        (PressureKind.CPU, PressureLevel.NORMAL),
        (PressureKind.MEMORY, PressureLevel.NORMAL),
        (PressureKind.GPU, PressureLevel.NORMAL),
    }


def test_fallbacks_are_considered_in_fixed_order_not_caller_order() -> None:
    offloaded = estimate(ram_gib=10, vram_mb=1_000)
    quantized = estimate(ram_gib=2, vram_mb=3_000)
    reduced = estimate(ram_gib=1, vram_mb=2_000)
    requested = estimate(
        ram_gib=2,
        vram_mb=20_000,
        alternatives=(
            ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),
            ResourceAlternative(FallbackAction.QUANTIZE, quantized),
            ResourceAlternative(FallbackAction.OFFLOAD, offloaded),
        ),
    )

    decision = broker().admit(requested)

    assert decision.admitted
    assert decision.action is FallbackAction.OFFLOAD
    assert decision.selected == offloaded
    assert decision.attempted_actions == (
        FallbackAction.NONE,
        FallbackAction.OFFLOAD,
    )


def test_quantize_follows_failed_offload_and_precedes_reduce_batch() -> None:
    offloaded = estimate(ram_gib=30, vram_mb=1_000)
    quantized = estimate(ram_gib=2, vram_mb=3_000)
    reduced = estimate(ram_gib=1, vram_mb=2_000)
    requested = estimate(
        vram_mb=20_000,
        alternatives=(
            ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),
            ResourceAlternative(FallbackAction.OFFLOAD, offloaded),
            ResourceAlternative(FallbackAction.QUANTIZE, quantized),
        ),
    )

    decision = broker().admit(requested)

    assert decision.action is FallbackAction.QUANTIZE
    assert decision.selected == quantized
    assert decision.attempted_actions == (
        FallbackAction.NONE,
        FallbackAction.OFFLOAD,
        FallbackAction.QUANTIZE,
    )


def test_sequence_precedes_batch_reduction_when_waiting_would_fit() -> None:
    instance = broker()
    first = instance.admit(estimate(vram_mb=5_000))
    assert first.admitted
    reduced = estimate(vram_mb=1_000)
    requested = estimate(
        vram_mb=5_000,
        alternatives=(ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),),
    )

    decision = instance.admit(requested)

    assert decision.outcome is AdmissionOutcome.DEFERRED
    assert decision.action is FallbackAction.SEQUENCE
    assert decision.selected == requested


def test_reduce_batch_is_used_when_sequence_cannot_solve_static_size() -> None:
    reduced = estimate(vram_mb=3_000)
    requested = estimate(
        vram_mb=20_000,
        alternatives=(ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),),
    )

    decision = broker().admit(requested)

    assert decision.admitted
    assert decision.action is FallbackAction.REDUCE_BATCH
    assert decision.selected == reduced
    assert decision.attempted_actions == (
        FallbackAction.NONE,
        FallbackAction.SEQUENCE,
        FallbackAction.REDUCE_BATCH,
    )


def test_rejection_is_explicit_and_lists_all_exceeded_budgets() -> None:
    decision = broker().admit(estimate(cpu_cores=15, ram_gib=29, vram_mb=20_000))

    assert decision.outcome is AdmissionOutcome.REJECTED
    assert decision.action is FallbackAction.REJECT
    assert decision.reservation is None
    assert decision.violations == (
        "cpu_cores",
        "ram_bytes",
        "gpu[0].vram_mb",
    )
    assert decision.attempted_actions[-1] is FallbackAction.REJECT
    assert tuple(item.code for item in decision.diagnostics) == (
        ResourceViolationCode.CPU_CORES,
        ResourceViolationCode.RAM_BYTES,
        ResourceViolationCode.GPU_VRAM,
    )
    assert tuple(item.transient for item in decision.diagnostics) == (
        False,
        False,
        False,
    )
    assert decision.diagnostics[0].requested == 15
    assert decision.diagnostics[0].available == 14


def test_deferred_decision_marks_concurrency_diagnostics_transient() -> None:
    instance = broker()
    first = instance.admit(estimate(vram_mb=2_000))
    assert first.admitted

    deferred = instance.admit(estimate(vram_mb=2_000))

    assert deferred.outcome is AdmissionOutcome.DEFERRED
    assert deferred.diagnostics
    assert all(item.transient for item in deferred.diagnostics)
    assert deferred.diagnostics[-1].code is ResourceViolationCode.GPU_HEAVY_JOBS


def test_unavailable_gpu_has_a_structured_non_transient_diagnostic() -> None:
    instance = broker()
    decision = instance.admit(
        ResourceEstimate(
            cpu_cores=1,
            ram_bytes=GIB,
            vram_mb=1,
            gpu_index=9,
            heavy_gpu=True,
        )
    )

    diagnostic = decision.diagnostics[-1]
    assert diagnostic.code is ResourceViolationCode.GPU_UNAVAILABLE
    assert diagnostic.path == "gpu[9].unavailable"
    assert diagnostic.gpu_index == 9
    assert diagnostic.available is None
    assert diagnostic.transient is False


def test_unknown_gpu_is_rejected_without_mutating_capacity() -> None:
    instance = broker()
    unknown = ResourceEstimate(
        cpu_cores=1,
        ram_bytes=GIB,
        vram_mb=1,
        gpu_index=9,
        heavy_gpu=True,
    )

    decision = instance.admit(unknown)

    assert decision.outcome is AdmissionOutcome.REJECTED
    assert decision.violations == ("gpu[9].unavailable",)
    assert instance.available == instance.limits


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cpu_cores": -1},
        {"ram_bytes": -1},
        {"vram_mb": -1},
        {"vram_mb": 1, "gpu_index": None},
        {"vram_mb": 0, "gpu_index": 0},
        {"heavy_gpu": True, "gpu_index": None},
    ),
)
def test_estimates_reject_incoherent_or_negative_values(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "cpu_cores": 1,
        "ram_bytes": GIB,
        "vram_mb": 0,
        "gpu_index": None,
        "heavy_gpu": False,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        ResourceEstimate(**values)  # type: ignore[arg-type]


def test_alternatives_cannot_repeat_an_action_or_nest() -> None:
    reduced = estimate(vram_mb=1_000)
    with pytest.raises(ValueError, match="duplicate fallback action"):
        ResourceEstimate(
            cpu_cores=1,
            ram_bytes=GIB,
            vram_mb=2_000,
            gpu_index=0,
            heavy_gpu=True,
            alternatives=(
                ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),
                ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),
            ),
        )

    nested = estimate(
        vram_mb=2_000,
        alternatives=(ResourceAlternative(FallbackAction.REDUCE_BATCH, reduced),),
    )
    with pytest.raises(ValueError, match="nested alternatives"):
        ResourceAlternative(FallbackAction.OFFLOAD, nested)

    with pytest.raises(ValueError, match="FallbackAction"):
        ResourceAlternative("offload", reduced)  # type: ignore[arg-type]
