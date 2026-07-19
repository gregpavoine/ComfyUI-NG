from __future__ import annotations

from dataclasses import dataclass


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ThreadBudget:
    python_threads: int
    omp_threads: int
    mkl_threads: int
    torch_threads: int
    torch_interop_threads: int

    def __post_init__(self) -> None:
        for name in (
            "python_threads",
            "omp_threads",
            "mkl_threads",
            "torch_threads",
            "torch_interop_threads",
        ):
            _positive_integer(name, getattr(self, name))

    def as_environment(self) -> dict[str, str]:
        """Return environment controls understood before native libraries import."""

        return {
            "OMP_NUM_THREADS": str(self.omp_threads),
            "MKL_NUM_THREADS": str(self.mkl_threads),
            "OPENBLAS_NUM_THREADS": str(self.omp_threads),
            "NUMEXPR_NUM_THREADS": str(self.omp_threads),
            "VECLIB_MAXIMUM_THREADS": str(self.omp_threads),
            "TORCH_NUM_THREADS": str(self.torch_threads),
        }


def compute_worker_count(
    *,
    physical_cores: int,
    reserve_cores: int,
    configured: int | None = None,
) -> int:
    """Bound configured workers to usable cores while retaining one worker."""

    _positive_integer("physical_cores", physical_cores)
    _non_negative_integer("reserve_cores", reserve_cores)
    if configured is not None:
        _positive_integer("configured", configured)
    usable = max(1, physical_cores - reserve_cores)
    return usable if configured is None else min(configured, usable)


@dataclass(frozen=True, slots=True)
class ThreadBudgetManager:
    physical_cores: int
    reserve_cores: int = 2
    max_threads_per_worker: int | None = None

    def __post_init__(self) -> None:
        _positive_integer("physical_cores", self.physical_cores)
        _non_negative_integer("reserve_cores", self.reserve_cores)
        if self.max_threads_per_worker is not None:
            _positive_integer("max_threads_per_worker", self.max_threads_per_worker)

    @property
    def available_cores(self) -> int:
        return max(1, self.physical_cores - self.reserve_cores)

    def allocate(self, *, worker_count: int) -> tuple[ThreadBudget, ...]:
        count = compute_worker_count(
            physical_cores=self.physical_cores,
            reserve_cores=self.reserve_cores,
            configured=worker_count,
        )
        quotient, remainder = divmod(self.available_cores, count)
        budgets: list[ThreadBudget] = []
        for index in range(count):
            assigned = quotient + (1 if index < remainder else 0)
            if self.max_threads_per_worker is not None:
                assigned = min(assigned, self.max_threads_per_worker)
            assigned = max(1, assigned)
            budgets.append(
                ThreadBudget(
                    python_threads=assigned,
                    omp_threads=assigned,
                    mkl_threads=assigned,
                    torch_threads=assigned,
                    torch_interop_threads=1,
                )
            )
        return tuple(budgets)


__all__ = ["ThreadBudget", "ThreadBudgetManager", "compute_worker_count"]
