from __future__ import annotations

import pytest

from comfyng.scheduler.priority import PriorityFactors
from comfyng.scheduler.queues import (
    QueueFullError,
    QueueItem,
    QueueName,
    SchedulerQueues,
)


def item(
    job_id: str,
    queue: QueueName,
    *,
    priority: int,
    enqueued_at: float,
) -> QueueItem:
    return QueueItem(
        job_id=job_id,
        queue=queue,
        enqueued_at=enqueued_at,
        factors=PriorityFactors(user_priority=priority),
    )


def test_exactly_six_named_queues_are_exposed() -> None:
    assert tuple(QueueName) == (
        QueueName.INTERACTIVE,
        QueueName.NORMAL,
        QueueName.BATCH,
        QueueName.BACKGROUND,
        QueueName.DOWNLOAD,
        QueueName.MAINTENANCE,
    )


def test_dynamic_age_prevents_starvation() -> None:
    queues = SchedulerQueues(max_queued=10, age_points_per_second=1)
    queues.enqueue(item("old-low", QueueName.BACKGROUND, priority=0, enqueued_at=0))
    queues.enqueue(
        item("new-high", QueueName.INTERACTIVE, priority=100, enqueued_at=150)
    )

    assert queues.pop_next(now=150).job_id == "old-low"


def test_ties_are_deterministic_by_queue_then_fifo() -> None:
    queues = SchedulerQueues(max_queued=10)
    queues.enqueue(item("normal-1", QueueName.NORMAL, priority=50, enqueued_at=10))
    queues.enqueue(
        item("interactive", QueueName.INTERACTIVE, priority=50, enqueued_at=10)
    )
    queues.enqueue(item("normal-2", QueueName.NORMAL, priority=50, enqueued_at=10))

    assert queues.pop_next(now=10).job_id == "interactive"
    assert queues.pop_next(now=10).job_id == "normal-1"
    assert queues.pop_next(now=10).job_id == "normal-2"


def test_global_and_per_queue_limits_raise_structured_backpressure() -> None:
    queues = SchedulerQueues(
        max_queued=2,
        per_queue_limits={QueueName.DOWNLOAD: 1},
    )
    queues.enqueue(item("d1", QueueName.DOWNLOAD, priority=1, enqueued_at=0))

    with pytest.raises(QueueFullError) as per_queue:
        queues.enqueue(item("d2", QueueName.DOWNLOAD, priority=1, enqueued_at=0))
    assert per_queue.value.queue is QueueName.DOWNLOAD
    assert per_queue.value.limit == 1
    assert per_queue.value.scope == "queue"

    queues.enqueue(item("n1", QueueName.NORMAL, priority=1, enqueued_at=0))
    with pytest.raises(QueueFullError) as global_limit:
        queues.enqueue(item("b1", QueueName.BATCH, priority=1, enqueued_at=0))
    assert global_limit.value.limit == 2
    assert global_limit.value.scope == "global"


def test_duplicate_ids_are_rejected_and_remove_is_idempotent() -> None:
    queues = SchedulerQueues(max_queued=5)
    queued = item("same", QueueName.NORMAL, priority=1, enqueued_at=0)
    queues.enqueue(queued)
    with pytest.raises(ValueError, match="already queued"):
        queues.enqueue(queued)

    assert queues.remove("same") == queued
    assert queues.remove("same") is None
    assert len(queues) == 0
