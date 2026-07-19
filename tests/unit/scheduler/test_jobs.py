from __future__ import annotations

import asyncio

import pytest

from comfyng.core.jobs import (
    InMemoryJobRepository,
    InvalidJobTransition,
    JobStatus,
    JobSubmission,
    JobTransitionConflict,
)
from comfyng.scheduler.queues import QueueName


def submission(job_id: str = "job-1", *, max_attempts: int = 3) -> JobSubmission:
    return JobSubmission(
        job_id=job_id,
        queue=QueueName.NORMAL.value,
        user_priority=50,
        payload={"prompt": "fixture"},
        max_attempts=max_attempts,
    )


def test_all_valid_lifecycle_transitions_are_monotonic() -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        record = await repository.create(submission(), now=1)
        assert record.status is JobStatus.QUEUED
        record = await repository.transition(
            record.job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.PREPARING,
            now=2,
        )
        record = await repository.transition(
            record.job_id,
            expected=JobStatus.PREPARING,
            target=JobStatus.RUNNING,
            now=3,
        )
        record = await repository.transition(
            record.job_id,
            expected=JobStatus.RUNNING,
            target=JobStatus.COMPLETED,
            now=4,
            result={"artifact": "cas://result"},
        )
        assert record.status is JobStatus.COMPLETED
        assert record.revision == 3
        assert record.started_at == 3
        assert record.finished_at == 4
        assert tuple(
            item.status for item in await repository.history(record.job_id)
        ) == (
            JobStatus.QUEUED,
            JobStatus.PREPARING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
        )

    asyncio.run(scenario())


def test_invalid_or_duplicate_terminal_transitions_are_rejected_atomically() -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        record = await repository.create(submission(), now=1)
        with pytest.raises(InvalidJobTransition):
            await repository.transition(
                record.job_id,
                expected=JobStatus.QUEUED,
                target=JobStatus.COMPLETED,
                now=2,
            )
        await repository.transition(
            record.job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.CANCELLED,
            now=3,
        )
        with pytest.raises(JobTransitionConflict):
            await repository.transition(
                record.job_id,
                expected=JobStatus.QUEUED,
                target=JobStatus.CANCELLED,
                now=4,
            )

    asyncio.run(scenario())


def test_concurrent_terminal_updates_have_exactly_one_winner() -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        record = await repository.create(submission(), now=1)

        results = await asyncio.gather(
            repository.transition(
                record.job_id,
                expected=JobStatus.QUEUED,
                target=JobStatus.FAILED,
                now=2,
            ),
            repository.transition(
                record.job_id,
                expected=JobStatus.QUEUED,
                target=JobStatus.CANCELLED,
                now=2,
            ),
            return_exceptions=True,
        )

        assert sum(not isinstance(item, BaseException) for item in results) == 1
        assert sum(isinstance(item, JobTransitionConflict) for item in results) == 1
        final = await repository.get(record.job_id)
        assert final is not None and final.status.terminal

    asyncio.run(scenario())


def test_retry_starts_a_new_monotonic_attempt_and_is_bounded() -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        record = await repository.create(submission(max_attempts=2), now=1)
        record = await repository.transition(
            record.job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.FAILED,
            now=2,
            error={"message": "failed"},
        )
        retried = await repository.retry(record.job_id, now=3)
        assert retried.status is JobStatus.QUEUED
        assert retried.attempt == 1
        assert retried.revision == 2
        await repository.transition(
            record.job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.FAILED,
            now=4,
        )
        with pytest.raises(InvalidJobTransition, match="retry budget"):
            await repository.retry(record.job_id, now=5)
        history = await repository.history(record.job_id)
        assert tuple(item.monotonic_position for item in history) == tuple(
            sorted(item.monotonic_position for item in history)
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("source", "target"),
    (
        (JobStatus.PREPARING, JobStatus.FAILED),
        (JobStatus.PREPARING, JobStatus.CANCELLED),
        (JobStatus.RUNNING, JobStatus.FAILED),
        (JobStatus.RUNNING, JobStatus.CANCELLED),
    ),
)
def test_every_non_happy_terminal_transition_is_supported(
    source: JobStatus,
    target: JobStatus,
) -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        record = await repository.create(submission(), now=1)
        record = await repository.transition(
            record.job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.PREPARING,
            now=2,
        )
        if source is JobStatus.RUNNING:
            record = await repository.transition(
                record.job_id,
                expected=JobStatus.PREPARING,
                target=JobStatus.RUNNING,
                now=3,
            )
        terminal = await repository.transition(
            record.job_id,
            expected=source,
            target=target,
            now=4,
        )
        assert terminal.status is target
        assert terminal.finished_at == 4

    asyncio.run(scenario())
