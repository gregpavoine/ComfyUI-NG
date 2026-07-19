from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

from .budgets import FallbackAction, ResourceEstimate, ResourceLimits
from .hardware import HardwareInventory
from .pressure import PressureEvent, PressureKind, PressureMonitor, PressureSample


class AdmissionOutcome(StrEnum):
    ADMITTED = "admitted"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class ResourceViolationCode(StrEnum):
    CPU_CORES = "cpu_cores"
    RAM_BYTES = "ram_bytes"
    PINNED_RAM_BYTES = "pinned_ram_bytes"
    CONCURRENT_READS = "concurrent_reads"
    CONCURRENT_WRITES = "concurrent_writes"
    GPU_UNAVAILABLE = "gpu_unavailable"
    GPU_VRAM = "gpu_vram_mb"
    GPU_HEAVY_JOBS = "gpu_heavy_jobs"


@dataclass(frozen=True, slots=True)
class ResourceViolation:
    code: ResourceViolationCode
    requested: int
    available: int | None
    transient: bool
    gpu_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, ResourceViolationCode):
            raise ValueError("code must be a ResourceViolationCode")
        if (
            isinstance(self.requested, bool)
            or not isinstance(self.requested, int)
            or self.requested < 0
        ):
            raise ValueError("requested must be a non-negative integer")
        if self.available is not None and (
            isinstance(self.available, bool)
            or not isinstance(self.available, int)
            or self.available < 0
        ):
            raise ValueError("available must be a non-negative integer or None")
        if not isinstance(self.transient, bool):
            raise ValueError("transient must be a boolean")
        if self.gpu_index is not None and (
            isinstance(self.gpu_index, bool)
            or not isinstance(self.gpu_index, int)
            or self.gpu_index < 0
        ):
            raise ValueError("gpu_index must be a non-negative integer or None")
        gpu_codes = {
            ResourceViolationCode.GPU_UNAVAILABLE,
            ResourceViolationCode.GPU_VRAM,
            ResourceViolationCode.GPU_HEAVY_JOBS,
        }
        if (self.code in gpu_codes) != (self.gpu_index is not None):
            raise ValueError("gpu_index presence must match the violation code")

    @property
    def path(self) -> str:
        if self.code is ResourceViolationCode.GPU_UNAVAILABLE:
            return f"gpu[{self.gpu_index}].unavailable"
        if self.code is ResourceViolationCode.GPU_VRAM:
            return f"gpu[{self.gpu_index}].vram_mb"
        if self.code is ResourceViolationCode.GPU_HEAVY_JOBS:
            return f"gpu[{self.gpu_index}].heavy_jobs"
        return self.code.value


class ResourceReservation:
    """An idempotently releasable, broker-owned resource lease."""

    __slots__ = ("_broker", "_released", "estimate", "reservation_id")

    def __init__(
        self,
        *,
        broker: ResourceBroker,
        reservation_id: UUID,
        estimate: ResourceEstimate,
    ) -> None:
        self._broker = broker
        self._released = False
        self.reservation_id = reservation_id
        self.estimate = estimate

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> bool:
        return self._broker.release(self)

    def _mark_released(self) -> None:
        self._released = True

    def __enter__(self) -> Self:
        if self.released:
            raise RuntimeError("cannot enter a released reservation")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    outcome: AdmissionOutcome
    action: FallbackAction
    requested: ResourceEstimate
    selected: ResourceEstimate
    diagnostics: tuple[ResourceViolation, ...]
    attempted_actions: tuple[FallbackAction, ...]
    reservation: ResourceReservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, AdmissionOutcome):
            raise ValueError("outcome must be an AdmissionOutcome")
        if not isinstance(self.action, FallbackAction):
            raise ValueError("action must be a FallbackAction")
        if not isinstance(self.requested, ResourceEstimate) or not isinstance(
            self.selected, ResourceEstimate
        ):
            raise ValueError("requested and selected must be ResourceEstimate values")
        if not isinstance(self.diagnostics, tuple) or not all(
            isinstance(item, ResourceViolation) for item in self.diagnostics
        ):
            raise ValueError("diagnostics must be a tuple of ResourceViolation values")
        if not isinstance(self.attempted_actions, tuple) or not all(
            isinstance(item, FallbackAction) for item in self.attempted_actions
        ):
            raise ValueError(
                "attempted_actions must be a tuple of FallbackAction values"
            )
        if self.outcome is AdmissionOutcome.ADMITTED and self.reservation is None:
            raise ValueError("an admitted decision requires a reservation")
        if (
            self.outcome is not AdmissionOutcome.ADMITTED
            and self.reservation is not None
        ):
            raise ValueError("a non-admitted decision cannot carry a reservation")
        if (
            self.outcome is AdmissionOutcome.REJECTED
            and self.action is not FallbackAction.REJECT
        ):
            raise ValueError("a rejected decision must use the reject action")
        if not self.attempted_actions:
            raise ValueError("attempted_actions cannot be empty")

    @property
    def admitted(self) -> bool:
        return self.outcome is AdmissionOutcome.ADMITTED

    @property
    def violations(self) -> tuple[str, ...]:
        """Stable path-only compatibility view over structured diagnostics."""

        return tuple(item.path for item in self.diagnostics)


_TRANSFORM_ORDER = (
    FallbackAction.OFFLOAD,
    FallbackAction.QUANTIZE,
)


class ResourceBroker:
    def __init__(
        self,
        *,
        inventory: HardwareInventory,
        reserve_cpu_cores: int = 2,
        reserve_ram_bytes: int = 4 * 1024**3,
        reserve_vram_mb: int = 768,
        max_pinned_ram_bytes: int = 8 * 1024**3,
        max_parallel_heavy_jobs: int = 1,
        max_concurrent_reads: int = 4,
        max_concurrent_writes: int = 2,
        pressure_monitor: PressureMonitor | None = None,
    ) -> None:
        self.inventory = inventory
        self.limits = ResourceLimits.from_inventory(
            inventory,
            reserve_cpu_cores=reserve_cpu_cores,
            reserve_ram_bytes=reserve_ram_bytes,
            reserve_vram_mb=reserve_vram_mb,
            max_pinned_ram_bytes=max_pinned_ram_bytes,
            max_parallel_heavy_jobs=max_parallel_heavy_jobs,
            max_concurrent_reads=max_concurrent_reads,
            max_concurrent_writes=max_concurrent_writes,
        )
        self._reservations: dict[UUID, ResourceReservation] = {}
        self._lock = RLock()
        self._pressure_monitor = pressure_monitor or PressureMonitor()
        self._pressure_events: list[PressureEvent] = []

    def _used(self) -> tuple[int, int, int, int, int, dict[int, int], dict[int, int]]:
        cpu = ram = pinned = reads = writes = 0
        vram = {index: 0 for index, _ in self.limits.vram_mb_by_gpu}
        heavy = {index: 0 for index, _ in self.limits.heavy_jobs_by_gpu}
        for reservation in self._reservations.values():
            item = reservation.estimate
            cpu += item.cpu_cores
            ram += item.ram_bytes
            pinned += item.pinned_ram_bytes
            reads += item.concurrent_reads
            writes += item.concurrent_writes
            if item.gpu_index is not None:
                vram[item.gpu_index] += item.vram_mb
                heavy[item.gpu_index] += int(item.heavy_gpu)
        return cpu, ram, pinned, reads, writes, vram, heavy

    @property
    def available(self) -> ResourceLimits:
        with self._lock:
            cpu, ram, pinned, reads, writes, vram, heavy = self._used()
            return ResourceLimits(
                cpu_cores=max(1, self.limits.cpu_cores - cpu)
                if self.limits.cpu_cores - cpu > 0
                else 0,
                ram_bytes=max(0, self.limits.ram_bytes - ram),
                pinned_ram_bytes=max(0, self.limits.pinned_ram_bytes - pinned),
                concurrent_reads=max(0, self.limits.concurrent_reads - reads),
                concurrent_writes=max(0, self.limits.concurrent_writes - writes),
                vram_mb_by_gpu=tuple(
                    (index, max(0, limit - vram[index]))
                    for index, limit in self.limits.vram_mb_by_gpu
                ),
                heavy_jobs_by_gpu=tuple(
                    (index, max(0, limit - heavy[index]))
                    for index, limit in self.limits.heavy_jobs_by_gpu
                ),
            )

    def _diagnostics(
        self,
        estimate: ResourceEstimate,
        *,
        against_current: bool,
    ) -> tuple[ResourceViolation, ...]:
        limits = self.available if against_current else self.limits
        diagnostics: list[ResourceViolation] = []

        def add(
            code: ResourceViolationCode,
            *,
            requested: int,
            available: int | None,
            static_available: int | None,
            gpu_index: int | None = None,
        ) -> None:
            diagnostics.append(
                ResourceViolation(
                    code=code,
                    requested=requested,
                    available=available,
                    transient=(
                        against_current
                        and static_available is not None
                        and requested <= static_available
                    ),
                    gpu_index=gpu_index,
                )
            )

        if estimate.cpu_cores > limits.cpu_cores:
            add(
                ResourceViolationCode.CPU_CORES,
                requested=estimate.cpu_cores,
                available=limits.cpu_cores,
                static_available=self.limits.cpu_cores,
            )
        if estimate.ram_bytes > limits.ram_bytes:
            add(
                ResourceViolationCode.RAM_BYTES,
                requested=estimate.ram_bytes,
                available=limits.ram_bytes,
                static_available=self.limits.ram_bytes,
            )
        if estimate.pinned_ram_bytes > limits.pinned_ram_bytes:
            add(
                ResourceViolationCode.PINNED_RAM_BYTES,
                requested=estimate.pinned_ram_bytes,
                available=limits.pinned_ram_bytes,
                static_available=self.limits.pinned_ram_bytes,
            )
        if estimate.concurrent_reads > limits.concurrent_reads:
            add(
                ResourceViolationCode.CONCURRENT_READS,
                requested=estimate.concurrent_reads,
                available=limits.concurrent_reads,
                static_available=self.limits.concurrent_reads,
            )
        if estimate.concurrent_writes > limits.concurrent_writes:
            add(
                ResourceViolationCode.CONCURRENT_WRITES,
                requested=estimate.concurrent_writes,
                available=limits.concurrent_writes,
                static_available=self.limits.concurrent_writes,
            )
        if estimate.gpu_index is not None:
            vram = limits.vram_mb(estimate.gpu_index)
            heavy = limits.heavy_jobs(estimate.gpu_index)
            static_vram = self.limits.vram_mb(estimate.gpu_index)
            static_heavy = self.limits.heavy_jobs(estimate.gpu_index)
            if vram is None or heavy is None:
                add(
                    ResourceViolationCode.GPU_UNAVAILABLE,
                    requested=estimate.vram_mb,
                    available=None,
                    static_available=None,
                    gpu_index=estimate.gpu_index,
                )
            else:
                if estimate.vram_mb > vram:
                    add(
                        ResourceViolationCode.GPU_VRAM,
                        requested=estimate.vram_mb,
                        available=vram,
                        static_available=static_vram,
                        gpu_index=estimate.gpu_index,
                    )
                if estimate.heavy_gpu and heavy < 1:
                    add(
                        ResourceViolationCode.GPU_HEAVY_JOBS,
                        requested=1,
                        available=heavy,
                        static_available=static_heavy,
                        gpu_index=estimate.gpu_index,
                    )
        return tuple(diagnostics)

    def _violations(
        self,
        estimate: ResourceEstimate,
        *,
        against_current: bool,
    ) -> tuple[str, ...]:
        return tuple(
            item.path
            for item in self._diagnostics(
                estimate,
                against_current=against_current,
            )
        )

    def _reserve(
        self,
        estimate: ResourceEstimate,
        *,
        reservation_id: UUID | None,
    ) -> ResourceReservation:
        identifier = reservation_id or uuid4()
        if identifier in self._reservations:
            raise ValueError(f"reservation already exists: {identifier}")
        reservation = ResourceReservation(
            broker=self,
            reservation_id=identifier,
            estimate=estimate,
        )
        self._reservations[identifier] = reservation
        self._observe_pressure()
        return reservation

    def _admitted(
        self,
        *,
        requested: ResourceEstimate,
        selected: ResourceEstimate,
        action: FallbackAction,
        attempts: list[FallbackAction],
        reservation_id: UUID | None,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            outcome=AdmissionOutcome.ADMITTED,
            action=action,
            requested=requested,
            selected=selected,
            diagnostics=(),
            attempted_actions=tuple(attempts),
            reservation=self._reserve(selected, reservation_id=reservation_id),
        )

    def admit(
        self,
        estimate: ResourceEstimate,
        *,
        reservation_id: UUID | None = None,
    ) -> AdmissionDecision:
        if not isinstance(estimate, ResourceEstimate):
            raise ValueError("estimate must be a ResourceEstimate")
        if reservation_id is not None and not isinstance(reservation_id, UUID):
            raise ValueError("reservation_id must be a UUID or None")
        with self._lock:
            attempts = [FallbackAction.NONE]
            sequence_candidate: ResourceEstimate | None = None
            current_violations = self._violations(estimate, against_current=True)
            if not current_violations:
                return self._admitted(
                    requested=estimate,
                    selected=estimate,
                    action=FallbackAction.NONE,
                    attempts=attempts,
                    reservation_id=reservation_id,
                )
            if not self._violations(estimate, against_current=False):
                sequence_candidate = estimate

            for action in _TRANSFORM_ORDER:
                alternative = estimate.alternative(action)
                if alternative is None:
                    continue
                attempts.append(action)
                if not self._violations(alternative, against_current=True):
                    return self._admitted(
                        requested=estimate,
                        selected=alternative,
                        action=action,
                        attempts=attempts,
                        reservation_id=reservation_id,
                    )
                if sequence_candidate is None and not self._violations(
                    alternative, against_current=False
                ):
                    sequence_candidate = alternative

            attempts.append(FallbackAction.SEQUENCE)
            if sequence_candidate is not None:
                return AdmissionDecision(
                    outcome=AdmissionOutcome.DEFERRED,
                    action=FallbackAction.SEQUENCE,
                    requested=estimate,
                    selected=sequence_candidate,
                    diagnostics=self._diagnostics(
                        sequence_candidate,
                        against_current=True,
                    ),
                    attempted_actions=tuple(attempts),
                )

            reduced = estimate.alternative(FallbackAction.REDUCE_BATCH)
            if reduced is not None:
                attempts.append(FallbackAction.REDUCE_BATCH)
                if not self._violations(reduced, against_current=True):
                    return self._admitted(
                        requested=estimate,
                        selected=reduced,
                        action=FallbackAction.REDUCE_BATCH,
                        attempts=attempts,
                        reservation_id=reservation_id,
                    )

            attempts.append(FallbackAction.REJECT)
            return AdmissionDecision(
                outcome=AdmissionOutcome.REJECTED,
                action=FallbackAction.REJECT,
                requested=estimate,
                selected=estimate,
                diagnostics=self._diagnostics(estimate, against_current=False),
                attempted_actions=tuple(attempts),
            )

    def release(self, reservation: ResourceReservation) -> bool:
        if not isinstance(reservation, ResourceReservation):
            raise ValueError("reservation must be a ResourceReservation")
        with self._lock:
            active = self._reservations.get(reservation.reservation_id)
            if active is not reservation:
                return False
            del self._reservations[reservation.reservation_id]
            reservation._mark_released()
            self._observe_pressure()
            return True

    def _observe_pressure(self) -> None:
        cpu, ram, pinned, reads, writes, vram, _heavy = self._used()
        samples = [
            (PressureKind.CPU, cpu, self.limits.cpu_cores, None),
            (PressureKind.MEMORY, ram, self.limits.ram_bytes, None),
            (
                PressureKind.PINNED_MEMORY,
                pinned,
                self.limits.pinned_ram_bytes,
                None,
            ),
            (PressureKind.IO_READ, reads, self.limits.concurrent_reads, None),
            (PressureKind.IO_WRITE, writes, self.limits.concurrent_writes, None),
        ]
        samples.extend(
            (PressureKind.GPU, vram[index], limit, index)
            for index, limit in self.limits.vram_mb_by_gpu
        )
        for kind, used, limit, device_index in samples:
            if limit < 1:
                continue
            event = self._pressure_monitor.observe(
                PressureSample(
                    kind=kind,
                    used=used,
                    limit=limit,
                    device_index=device_index,
                )
            )
            if event is not None:
                self._pressure_events.append(event)

    def drain_pressure_events(self) -> tuple[PressureEvent, ...]:
        with self._lock:
            events = tuple(self._pressure_events)
            self._pressure_events.clear()
            return events

    @property
    def active_reservations(self) -> tuple[ResourceReservation, ...]:
        with self._lock:
            return tuple(self._reservations.values())


__all__ = [
    "AdmissionDecision",
    "AdmissionOutcome",
    "FallbackAction",
    "ResourceBroker",
    "ResourceReservation",
    "ResourceViolation",
    "ResourceViolationCode",
]
