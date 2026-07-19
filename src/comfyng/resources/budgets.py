from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .hardware import HardwareInventory


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class FallbackAction(StrEnum):
    NONE = "none"
    OFFLOAD = "offload"
    QUANTIZE = "quantize"
    SEQUENCE = "sequence"
    REDUCE_BATCH = "reduce_batch"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    cpu_cores: int
    ram_bytes: int
    vram_mb: int = 0
    gpu_index: int | None = None
    heavy_gpu: bool = False
    pinned_ram_bytes: int = 0
    concurrent_reads: int = 0
    concurrent_writes: int = 0
    alternatives: tuple[ResourceAlternative, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "cpu_cores",
            "ram_bytes",
            "vram_mb",
            "pinned_ram_bytes",
            "concurrent_reads",
            "concurrent_writes",
        ):
            _non_negative_integer(name, getattr(self, name))
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be >= 1")
        if self.gpu_index is not None:
            _non_negative_integer("gpu_index", self.gpu_index)
        if not isinstance(self.heavy_gpu, bool):
            raise ValueError("heavy_gpu must be a boolean")
        if (self.vram_mb > 0) != (self.gpu_index is not None):
            raise ValueError("vram_mb and gpu_index must be specified together")
        if self.heavy_gpu and self.gpu_index is None:
            raise ValueError("a heavy GPU estimate requires gpu_index")
        if self.pinned_ram_bytes > self.ram_bytes:
            raise ValueError("pinned_ram_bytes cannot exceed ram_bytes")
        if not isinstance(self.alternatives, tuple) or not all(
            isinstance(item, ResourceAlternative) for item in self.alternatives
        ):
            raise ValueError(
                "alternatives must be a tuple of ResourceAlternative values"
            )
        actions = tuple(item.action for item in self.alternatives)
        if len(actions) != len(set(actions)):
            raise ValueError("duplicate fallback action")

    def alternative(self, action: FallbackAction) -> ResourceEstimate | None:
        return next(
            (item.estimate for item in self.alternatives if item.action is action),
            None,
        )


@dataclass(frozen=True, slots=True)
class ResourceAlternative:
    action: FallbackAction
    estimate: ResourceEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.action, FallbackAction):
            raise ValueError("action must be a FallbackAction")
        if self.action not in {
            FallbackAction.OFFLOAD,
            FallbackAction.QUANTIZE,
            FallbackAction.REDUCE_BATCH,
        }:
            raise ValueError("fallback alternative action is not transformable")
        if not isinstance(self.estimate, ResourceEstimate):
            raise ValueError("estimate must be a ResourceEstimate")
        if self.estimate.alternatives:
            raise ValueError("nested alternatives are not allowed")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    cpu_cores: int
    ram_bytes: int
    vram_mb_by_gpu: tuple[tuple[int, int], ...] = ()
    pinned_ram_bytes: int = 0
    concurrent_reads: int = 4
    concurrent_writes: int = 2
    heavy_jobs_by_gpu: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "cpu_cores",
            "ram_bytes",
            "pinned_ram_bytes",
            "concurrent_reads",
            "concurrent_writes",
        ):
            _non_negative_integer(name, getattr(self, name))
        self._validate_pairs("vram_mb_by_gpu", self.vram_mb_by_gpu)
        self._validate_pairs("heavy_jobs_by_gpu", self.heavy_jobs_by_gpu)
        if tuple(index for index, _ in self.vram_mb_by_gpu) != tuple(
            index for index, _ in self.heavy_jobs_by_gpu
        ):
            raise ValueError("GPU resource maps must contain the same indexes")

    @staticmethod
    def _validate_pairs(name: str, values: tuple[tuple[int, int], ...]) -> None:
        if not isinstance(values, tuple):
            raise ValueError(f"{name} must be a tuple")
        indexes: list[int] = []
        for item in values:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"{name} entries must be (index, value) pairs")
            index, value = item
            _non_negative_integer(f"{name}.index", index)
            _non_negative_integer(f"{name}.value", value)
            indexes.append(index)
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise ValueError(f"{name} indexes must be sorted and unique")

    @classmethod
    def from_inventory(
        cls,
        inventory: HardwareInventory,
        *,
        reserve_cpu_cores: int,
        reserve_ram_bytes: int,
        reserve_vram_mb: int,
        max_pinned_ram_bytes: int = 8 * 1024**3,
        max_parallel_heavy_jobs: int = 1,
        max_concurrent_reads: int = 4,
        max_concurrent_writes: int = 2,
    ) -> ResourceLimits:
        if not isinstance(inventory, HardwareInventory):
            raise ValueError("inventory must be a HardwareInventory")
        for name, value in (
            ("reserve_cpu_cores", reserve_cpu_cores),
            ("reserve_ram_bytes", reserve_ram_bytes),
            ("reserve_vram_mb", reserve_vram_mb),
            ("max_pinned_ram_bytes", max_pinned_ram_bytes),
            ("max_parallel_heavy_jobs", max_parallel_heavy_jobs),
            ("max_concurrent_reads", max_concurrent_reads),
            ("max_concurrent_writes", max_concurrent_writes),
        ):
            _non_negative_integer(name, value)
        if max_parallel_heavy_jobs < 1:
            raise ValueError("max_parallel_heavy_jobs must be >= 1")
        cpu = max(1, inventory.cpu.physical_cores - reserve_cpu_cores)
        total_ram_budget = max(0, inventory.memory.total_bytes - reserve_ram_bytes)
        live_ram_budget = max(0, inventory.memory.available_bytes - reserve_ram_bytes)
        ram = min(total_ram_budget, live_ram_budget)
        pinned = min(max_pinned_ram_bytes, ram)
        vram: list[tuple[int, int]] = []
        heavy: list[tuple[int, int]] = []
        for gpu in inventory.gpus:
            total_budget = max(0, gpu.total_vram_mb - reserve_vram_mb)
            live_budget = max(0, gpu.free_vram_mb - reserve_vram_mb)
            vram.append((gpu.index, min(total_budget, live_budget)))
            heavy.append((gpu.index, max_parallel_heavy_jobs))
        return cls(
            cpu_cores=cpu,
            ram_bytes=ram,
            vram_mb_by_gpu=tuple(vram),
            pinned_ram_bytes=pinned,
            concurrent_reads=max_concurrent_reads,
            concurrent_writes=max_concurrent_writes,
            heavy_jobs_by_gpu=tuple(heavy),
        )

    def vram_mb(self, gpu_index: int) -> int | None:
        return next(
            (value for index, value in self.vram_mb_by_gpu if index == gpu_index),
            None,
        )

    def heavy_jobs(self, gpu_index: int) -> int | None:
        return next(
            (value for index, value in self.heavy_jobs_by_gpu if index == gpu_index),
            None,
        )


__all__ = [
    "FallbackAction",
    "ResourceAlternative",
    "ResourceEstimate",
    "ResourceLimits",
]
