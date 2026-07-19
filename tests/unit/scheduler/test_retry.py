from __future__ import annotations

import pytest

from comfyng.scheduler.retry import RetryPolicy


class TemporaryFailure(RuntimeError):
    pass


def test_retry_is_bounded_and_exponential_delay_is_capped() -> None:
    policy = RetryPolicy(
        max_attempts=4,
        base_delay_seconds=1,
        max_delay_seconds=2.5,
        retryable_exceptions=(TemporaryFailure,),
    )

    assert policy.should_retry(attempt=0, error=TemporaryFailure())
    assert policy.should_retry(attempt=1, error=TemporaryFailure())
    assert policy.should_retry(attempt=2, error=TemporaryFailure())
    assert not policy.should_retry(attempt=3, error=TemporaryFailure())
    assert policy.delay_for_retry(next_attempt=1) == 1
    assert policy.delay_for_retry(next_attempt=2) == 2
    assert policy.delay_for_retry(next_attempt=3) == 2.5


def test_non_retryable_errors_are_never_retried() -> None:
    policy = RetryPolicy(
        max_attempts=3,
        retryable_exceptions=(TemporaryFailure,),
    )
    assert not policy.should_retry(attempt=0, error=ValueError("permanent"))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_attempts": 0},
        {"base_delay_seconds": -1},
        {"max_delay_seconds": -1},
        {"base_delay_seconds": 2, "max_delay_seconds": 1},
    ),
)
def test_retry_policy_rejects_invalid_bounds(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]
