from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
from threading import RLock

from .priority import PriorityFactors, priority_score, queue_age_bonus


class QueueName(StrEnum):
    INTERACTIVE = "interactive"
    NORMAL = "normal"
    BATCH = "batch"
    BACKGROUND = "background"
    DOWNLOAD = "download"
    MAINTENANCE = "maintenance"


_QUEUE_ORDER = {queue: index for index, queue in enumerate(QueueName)}


class QueueFullError(RuntimeError):
    def __init__(
        self,
        *,
        queue: QueueName,
        current: int,
        limit: int,
        scope: str,
    ) -> None:
        self.queue = queue
        self.current = current
        self.limit = limit
        self.scope = scope
        super().__init__(
            f"{scope} queue capacity reached for {queue.value}: {current}/{limit}"
        )


@dataclass(frozen=True, slots=True)
class QueueItem:
    job_id: str
    queue: QueueName
    enqueued_at: float
    factors: PriorityFactors

    def __post_init__(self) -> None:
        if (
            not isinstance(self.job_id, str)
            or not self.job_id
            or self.job_id != self.job_id.strip()
        ):
            raise ValueError("job_id must be a non-empty trimmed string")
        if not isinstance(self.queue, QueueName):
            raise ValueError("queue must be a QueueName")
        if (
            isinstance(self.enqueued_at, bool)
            or not isinstance(self.enqueued_at, (int, float))
            or not math.isfinite(self.enqueued_at)
            or self.enqueued_at < 0
        ):
            raise ValueError("enqueued_at must be a finite non-negative number")
        if not isinstance(self.factors, PriorityFactors):
            raise ValueError("factors must be PriorityFactors")


@dataclass(frozen=True, slots=True)
class _QueuedItem:
    item: QueueItem
    sequence: int


class SchedulerQueues:
    def __init__(
        self,
        *,
        max_queued: int,
        per_queue_limits: Mapping[QueueName, int] | None = None,
        age_points_per_second: float = 1.0,
    ) -> None:
        if (
            isinstance(max_queued, bool)
            or not isinstance(max_queued, int)
            or max_queued < 1
        ):
            raise ValueError("max_queued must be a positive integer")
        if (
            isinstance(age_points_per_second, bool)
            or not isinstance(age_points_per_second, (int, float))
            or not math.isfinite(age_points_per_second)
            or age_points_per_second < 0
        ):
            raise ValueError("age_points_per_second must be finite and non-negative")
        limits = dict(per_queue_limits or {})
        for queue, limit in limits.items():
            if not isinstance(queue, QueueName):
                raise ValueError("per_queue_limits keys must be QueueName values")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("per-queue limits must be positive integers")
        self.max_queued = max_queued
        self.per_queue_limits = limits
        self.age_points_per_second = float(age_points_per_second)
        self._items: dict[str, _QueuedItem] = {}
        self._by_queue: dict[QueueName, int] = {queue: 0 for queue in QueueName}
        self._next_sequence = 1
        self._lock = RLock()

    def ensure_capacity(self, queue: QueueName) -> None:
        if not isinstance(queue, QueueName):
            raise ValueError("queue must be a QueueName")
        with self._lock:
            if len(self._items) >= self.max_queued:
                raise QueueFullError(
                    queue=queue,
                    current=len(self._items),
                    limit=self.max_queued,
                    scope="global",
                )
            limit = self.per_queue_limits.get(queue)
            if limit is not None and self._by_queue[queue] >= limit:
                raise QueueFullError(
                    queue=queue,
                    current=self._by_queue[queue],
                    limit=limit,
                    scope="queue",
                )

    def enqueue(self, item: QueueItem) -> None:
        if not isinstance(item, QueueItem):
            raise ValueError("item must be a QueueItem")
        with self._lock:
            if item.job_id in self._items:
                raise ValueError(f"job {item.job_id!r} is already queued")
            self.ensure_capacity(item.queue)
            self._items[item.job_id] = _QueuedItem(item, self._next_sequence)
            self._next_sequence += 1
            self._by_queue[item.queue] += 1

    def _score(self, queued: _QueuedItem, now: float) -> float:
        item = queued.item
        factors = item.factors
        return priority_score(
            PriorityFactors(
                user_priority=factors.user_priority,
                queue_age_bonus=factors.queue_age_bonus
                + queue_age_bonus(
                    enqueued_at=item.enqueued_at,
                    now=now,
                    points_per_second=self.age_points_per_second,
                ),
                warm_model_bonus=factors.warm_model_bonus,
                cache_reuse_bonus=factors.cache_reuse_bonus,
                memory_pressure_penalty=factors.memory_pressure_penalty,
                estimated_duration_penalty=factors.estimated_duration_penalty,
            )
        )

    def pop_next(self, *, now: float) -> QueueItem:
        with self._lock:
            if not self._items:
                raise IndexError("scheduler queues are empty")
            selected = min(
                self._items.values(),
                key=lambda queued: (
                    -self._score(queued, now),
                    _QUEUE_ORDER[queued.item.queue],
                    queued.item.enqueued_at,
                    queued.sequence,
                    queued.item.job_id,
                ),
            )
            del self._items[selected.item.job_id]
            self._by_queue[selected.item.queue] -= 1
            return selected.item

    def remove(self, job_id: str) -> QueueItem | None:
        with self._lock:
            queued = self._items.pop(job_id, None)
            if queued is None:
                return None
            self._by_queue[queued.item.queue] -= 1
            return queued.item

    def contains(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshot(self) -> tuple[QueueItem, ...]:
        with self._lock:
            return tuple(value.item for value in self._items.values())


__all__ = [
    "QueueFullError",
    "QueueItem",
    "QueueName",
    "SchedulerQueues",
]
