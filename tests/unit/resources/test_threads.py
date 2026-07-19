from __future__ import annotations

import pytest

from comfyng.resources.threads import (
    ThreadBudget,
    ThreadBudgetManager,
    compute_worker_count,
)


def test_compute_worker_count_reserves_cores_and_honours_user_bound() -> None:
    assert compute_worker_count(physical_cores=16, reserve_cores=2) == 14
    assert (
        compute_worker_count(
            physical_cores=16,
            reserve_cores=2,
            configured=6,
        )
        == 6
    )


def test_compute_worker_count_never_falls_below_one() -> None:
    assert compute_worker_count(physical_cores=2, reserve_cores=2) == 1
    assert compute_worker_count(physical_cores=1, reserve_cores=8) == 1


def test_thread_budgets_are_bounded_by_available_physical_cores() -> None:
    manager = ThreadBudgetManager(physical_cores=10, reserve_cores=2)

    budgets = manager.allocate(worker_count=3)

    assert tuple(item.python_threads for item in budgets) == (3, 3, 2)
    assert sum(item.python_threads for item in budgets) == 8
    assert all(item.omp_threads == item.python_threads for item in budgets)
    assert all(item.torch_interop_threads == 1 for item in budgets)


def test_more_workers_than_cores_are_capped_without_zero_thread_budgets() -> None:
    manager = ThreadBudgetManager(physical_cores=6, reserve_cores=2)

    budgets = manager.allocate(worker_count=20)

    assert len(budgets) == 4
    assert all(item.python_threads == 1 for item in budgets)


def test_max_threads_per_worker_leaves_excess_capacity_unassigned() -> None:
    manager = ThreadBudgetManager(
        physical_cores=32,
        reserve_cores=2,
        max_threads_per_worker=4,
    )

    budgets = manager.allocate(worker_count=3)

    assert tuple(item.python_threads for item in budgets) == (4, 4, 4)


def test_thread_environment_controls_all_native_pools() -> None:
    budget = ThreadBudget(
        python_threads=4,
        omp_threads=3,
        mkl_threads=2,
        torch_threads=4,
        torch_interop_threads=1,
    )

    assert budget.as_environment() == {
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "3",
        "VECLIB_MAXIMUM_THREADS": "3",
        "TORCH_NUM_THREADS": "4",
    }


def test_thread_budget_rejects_zero_and_boolean_values() -> None:
    with pytest.raises(ValueError):
        ThreadBudget(
            python_threads=0,
            omp_threads=1,
            mkl_threads=1,
            torch_threads=1,
            torch_interop_threads=1,
        )
    with pytest.raises(ValueError):
        ThreadBudget(
            python_threads=True,  # type: ignore[arg-type]
            omp_threads=1,
            mkl_threads=1,
            torch_threads=1,
            torch_interop_threads=1,
        )


@pytest.mark.parametrize(
    ("physical", "reserve", "configured"),
    ((0, 0, None), (4, -1, None), (4, 1, 0), (True, 1, None)),
)
def test_worker_count_rejects_invalid_inputs(
    physical: int,
    reserve: int,
    configured: int | None,
) -> None:
    with pytest.raises(ValueError):
        compute_worker_count(
            physical_cores=physical,
            reserve_cores=reserve,
            configured=configured,
        )
