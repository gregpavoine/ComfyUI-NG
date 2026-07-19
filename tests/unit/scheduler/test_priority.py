from __future__ import annotations

import math

import pytest

from comfyng.scheduler.priority import PriorityFactors, queue_age_bonus, priority_score


def test_priority_formula_is_exact() -> None:
    factors = PriorityFactors(
        user_priority=80,
        queue_age_bonus=7.5,
        warm_model_bonus=11,
        cache_reuse_bonus=13,
        memory_pressure_penalty=17,
        estimated_duration_penalty=19.5,
    )

    assert priority_score(factors) == 80 + 7.5 + 11 + 13 - 17 - 19.5


def test_queue_age_bonus_is_linear_unbounded_and_never_negative() -> None:
    assert queue_age_bonus(enqueued_at=10, now=5, points_per_second=2) == 0
    assert queue_age_bonus(enqueued_at=10, now=13, points_per_second=2) == 6
    assert (
        queue_age_bonus(enqueued_at=0, now=1_000_000, points_per_second=1) == 1_000_000
    )


def test_priority_rejects_non_finite_or_out_of_range_contract_values() -> None:
    with pytest.raises(ValueError):
        PriorityFactors(user_priority=101)
    with pytest.raises(ValueError):
        PriorityFactors(user_priority=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PriorityFactors(user_priority=50, queue_age_bonus=math.inf)
    with pytest.raises(ValueError):
        queue_age_bonus(enqueued_at=0, now=1, points_per_second=-1)
