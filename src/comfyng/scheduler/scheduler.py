from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import inspect
import math
import time
from typing import Any, Protocol

from comfyng.core.cache import NodeResultCache
from comfyng.core.jobs import (
    JobRecord,
    JobRepository,
    JobStatus,
    JobSubmission,
)
from comfyng.core.json_values import freeze_json_value
from comfyng.events.bus import EventBus
from comfyng.resources.broker import AdmissionOutcome

from .cancellation import (
    CancellationCheckpoint,
    CancellationRequested,
    CancellationToken,
)
from .priority import PriorityFactors
from .queues import QueueFullError, QueueItem, QueueName, SchedulerQueues
from .retry import RetryPolicy


@dataclass(frozen=True, slots=True)
class DispatchResult:
    value: Any
    cacheable: bool = False
    size_bytes: int = 0
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cacheable, bool):
            raise ValueError("cacheable must be a boolean")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer")
        if self.ttl_seconds is not None and (
            isinstance(self.ttl_seconds, bool)
            or not isinstance(self.ttl_seconds, (int, float))
            or not math.isfinite(self.ttl_seconds)
            or self.ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be finite and positive or None")
        object.__setattr__(self, "value", freeze_json_value(self.value, path="$.value"))


class WorkerDispatcher(Protocol):
    async def dispatch(
        self,
        job: JobRecord,
        token: CancellationToken,
    ) -> DispatchResult: ...

    async def cancel(self, job_id: str) -> None: ...


class AdmissionBroker(Protocol):
    def admit(self, estimate: Any) -> Any: ...


class SchedulerBackpressure(RuntimeError):
    code = "QUEUE_FULL"

    def __init__(self, error: QueueFullError) -> None:
        self.queue = error.queue
        self.current = error.current
        self.limit = error.limit
        self.scope = error.scope
        super().__init__(str(error))


class ResourceAdmissionFailure(RuntimeError):
    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__(
            "resource admission rejected"
            + (f": {', '.join(violations)}" if violations else "")
        )


def _error_payload(error: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    violations = getattr(error, "violations", None)
    if isinstance(violations, tuple):
        payload["violations"] = list(violations)
    return payload


class Scheduler:
    def __init__(
        self,
        *,
        repository: JobRepository,
        events: EventBus,
        cache: NodeResultCache,
        broker: AdmissionBroker,
        dispatcher: WorkerDispatcher,
        retry_policy: RetryPolicy,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_queued: int = 100,
        per_queue_limits: Mapping[QueueName, int] | None = None,
        max_concurrency: int = 4,
        age_points_per_second: float = 1.0,
        resource_poll_interval: float = 0.05,
    ) -> None:
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if (
            isinstance(resource_poll_interval, bool)
            or not isinstance(resource_poll_interval, (int, float))
            or not math.isfinite(resource_poll_interval)
            or resource_poll_interval <= 0
        ):
            raise ValueError("resource_poll_interval must be finite and positive")
        self.repository = repository
        self.events = events
        self.cache = cache
        self.broker = broker
        self.dispatcher = dispatcher
        self.retry_policy = retry_policy
        self._clock = clock
        self._sleeper = sleeper
        self.max_concurrency = max_concurrency
        self.resource_poll_interval = float(resource_poll_interval)
        self.queues = SchedulerQueues(
            max_queued=max_queued,
            per_queue_limits=per_queue_limits,
            age_points_per_second=age_points_per_second,
        )
        self._active: dict[str, asyncio.Task[None]] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._reservations: dict[str, Any] = {}
        self._control_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._stop_requested = False

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("scheduler clock must return a finite non-negative number")
        return float(value)

    @staticmethod
    def _factors(record: JobRecord | JobSubmission) -> PriorityFactors:
        return PriorityFactors(
            user_priority=record.user_priority,
            warm_model_bonus=record.warm_model_bonus,
            cache_reuse_bonus=record.cache_reuse_bonus,
            memory_pressure_penalty=record.memory_pressure_penalty,
            estimated_duration_penalty=record.estimated_duration_penalty,
        )

    @staticmethod
    def _queue(value: str) -> QueueName:
        try:
            return QueueName(value)
        except ValueError as exc:
            raise ValueError(f"unknown scheduler queue: {value!r}") from exc

    async def submit(self, submission: JobSubmission) -> JobRecord:
        if not isinstance(submission, JobSubmission):
            raise ValueError("submission must be JobSubmission")
        queue = self._queue(submission.queue)
        async with self._control_lock:
            try:
                self.queues.ensure_capacity(queue)
            except QueueFullError as exc:
                raise SchedulerBackpressure(exc) from exc
            now = self._now()
            record = await self.repository.create(submission, now=now)
            try:
                self.queues.enqueue(
                    QueueItem(
                        job_id=record.job_id,
                        queue=queue,
                        enqueued_at=record.queued_at,
                        factors=self._factors(record),
                    )
                )
            except BaseException:
                # Capacity was checked while holding the same lock, so this can only
                # be an invariant violation or duplicate ID and must stay visible.
                raise
            await self.events.publish(
                "job.created",
                {"status": record.status.value, "attempt": record.attempt},
                job_id=record.job_id,
            )
            await self.events.publish(
                "job.queued",
                {"queue": record.queue, "priority": record.user_priority},
                job_id=record.job_id,
            )
            self._wake.set()
            return record

    async def _transition(
        self,
        job_id: str,
        *,
        expected: JobStatus,
        target: JobStatus,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
        event_payload: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        record = await self.repository.transition(
            job_id,
            expected=expected,
            target=target,
            now=self._now(),
            result=result,
            error=error,
        )
        payload = {
            "status": record.status.value,
            "attempt": record.attempt,
            "revision": record.revision,
        }
        if event_payload:
            payload.update(event_payload)
        await self.events.publish(
            f"job.{target.value}",
            payload,
            job_id=job_id,
        )
        return record

    async def _complete_from_cache(self, record: JobRecord, value: Any) -> None:
        record = await self._transition(
            record.job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.PREPARING,
            event_payload={"cache_hit": True},
        )
        record = await self._transition(
            record.job_id,
            expected=JobStatus.PREPARING,
            target=JobStatus.RUNNING,
            event_payload={"cache_hit": True},
        )
        await self._transition(
            record.job_id,
            expected=JobStatus.RUNNING,
            target=JobStatus.COMPLETED,
            result=value,
            event_payload={"cache_hit": True},
        )

    async def _launch_next(self) -> bool:
        async with self._control_lock:
            if len(self._active) >= self.max_concurrency or len(self.queues) == 0:
                return False
            item = self.queues.pop_next(now=self._now())
            record = await self.repository.get(item.job_id)
            if record is None or record.status is not JobStatus.QUEUED:
                return True

            if record.cache_key is not None:
                try:
                    cached = await self.cache.get(record.cache_key)
                except Exception as cache_error:
                    cached = None
                    await self.events.publish(
                        "cache.read_failed",
                        {
                            "cache_key": record.cache_key,
                            "error": _error_payload(cache_error),
                        },
                        job_id=record.job_id,
                    )
                if cached is not None:
                    await self._complete_from_cache(record, cached.value)
                    return True

            reservation = None
            if record.resource_estimate is not None:
                try:
                    decision = self.broker.admit(record.resource_estimate)
                except Exception as broker_error:
                    await self._transition(
                        record.job_id,
                        expected=JobStatus.QUEUED,
                        target=JobStatus.FAILED,
                        error=_error_payload(broker_error),
                        event_payload={"phase": "resource_broker"},
                    )
                    return True
                if not bool(getattr(decision, "admitted", False)):
                    outcome = getattr(decision, "outcome", None)
                    if (
                        outcome is AdmissionOutcome.DEFERRED
                        or str(outcome) == "deferred"
                    ):
                        self.queues.enqueue(item)
                        await self.events.publish(
                            "job.deferred",
                            {
                                "reason": "resources",
                                "violations": list(getattr(decision, "violations", ())),
                            },
                            job_id=record.job_id,
                        )
                        return False
                    error = ResourceAdmissionFailure(
                        tuple(getattr(decision, "violations", ()))
                    )
                    await self._transition(
                        record.job_id,
                        expected=JobStatus.QUEUED,
                        target=JobStatus.FAILED,
                        error=_error_payload(error),
                        event_payload={"phase": "resource_admission"},
                    )
                    return True
                reservation = getattr(decision, "reservation", None)
                if reservation is None:
                    raise RuntimeError("admitted resource decision has no reservation")

            record = await self._transition(
                record.job_id,
                expected=JobStatus.QUEUED,
                target=JobStatus.PREPARING,
            )
            token = CancellationToken()
            self._tokens[record.job_id] = token
            if reservation is not None:
                self._reservations[record.job_id] = reservation
            task = asyncio.create_task(
                self._execute(record.job_id, token),
                name=f"comfyng-job-{record.job_id}",
            )
            self._active[record.job_id] = task
            return True

    async def _release_reservation(self, job_id: str) -> None:
        reservation = self._reservations.pop(job_id, None)
        if reservation is None:
            return
        released = reservation.release()
        if inspect.isawaitable(released):
            await released

    async def _execute(self, job_id: str, token: CancellationToken) -> None:
        try:
            record = await self._transition(
                job_id,
                expected=JobStatus.PREPARING,
                target=JobStatus.RUNNING,
            )
            token.checkpoint(CancellationCheckpoint.BETWEEN_BLOCKS)
            result = await self.dispatcher.dispatch(record, token)
            if not isinstance(result, DispatchResult):
                raise TypeError("dispatcher must return DispatchResult")
            token.checkpoint(CancellationCheckpoint.BEFORE_SAVE)
            if result.cacheable and record.cache_key is not None:
                try:
                    await self.cache.put(
                        record.cache_key,
                        result.value,
                        size_bytes=result.size_bytes,
                        ttl_seconds=result.ttl_seconds,
                    )
                except Exception as cache_error:
                    await self.events.publish(
                        "cache.write_failed",
                        {
                            "cache_key": record.cache_key,
                            "error": _error_payload(cache_error),
                        },
                        job_id=job_id,
                    )
            await self._transition(
                job_id,
                expected=JobStatus.RUNNING,
                target=JobStatus.COMPLETED,
                result=result.value,
                event_payload={"cache_hit": False},
            )
        except (CancellationRequested, asyncio.CancelledError) as exc:
            record = await self.repository.get(job_id)
            if record is not None and not record.status.terminal:
                reason = token.reason or str(exc) or "cancelled"
                await self._transition(
                    job_id,
                    expected=record.status,
                    target=JobStatus.CANCELLED,
                    error={"reason": reason},
                )
        except Exception as exc:
            record = await self.repository.get(job_id)
            if record is None or record.status.terminal:
                return
            can_retry = (
                record.attempt + 1 < record.max_attempts
                and self.retry_policy.should_retry(attempt=record.attempt, error=exc)
            )
            if can_retry:
                await self._release_reservation(job_id)
                delay = self.retry_policy.delay_for_retry(
                    next_attempt=record.attempt + 1
                )
                await self.events.publish(
                    "job.retrying",
                    {
                        "attempt": record.attempt,
                        "next_attempt": record.attempt + 1,
                        "delay_seconds": delay,
                        "error": _error_payload(exc),
                    },
                    job_id=job_id,
                )
                try:
                    await self._sleeper(delay)
                except asyncio.CancelledError:
                    token.cancel("scheduler task cancelled")
                if token.cancelled:
                    current = await self.repository.get(job_id)
                    if current is not None and not current.status.terminal:
                        await self._transition(
                            job_id,
                            expected=current.status,
                            target=JobStatus.CANCELLED,
                            error={"reason": token.reason or "cancelled"},
                        )
                    return
                async with self._control_lock:
                    try:
                        self.queues.ensure_capacity(self._queue(record.queue))
                    except QueueFullError as queue_error:
                        await self._transition(
                            job_id,
                            expected=record.status,
                            target=JobStatus.FAILED,
                            error=_error_payload(SchedulerBackpressure(queue_error)),
                            event_payload={"phase": "retry_backpressure"},
                        )
                        return
                    retried = await self.repository.retry(
                        job_id,
                        expected=record.status,
                        now=self._now(),
                        error=_error_payload(exc),
                    )
                    self.queues.enqueue(
                        QueueItem(
                            job_id=job_id,
                            queue=self._queue(retried.queue),
                            enqueued_at=retried.queued_at,
                            factors=self._factors(retried),
                        )
                    )
                    await self.events.publish(
                        "job.queued",
                        {"queue": retried.queue, "attempt": retried.attempt},
                        job_id=job_id,
                    )
                    self._wake.set()
            else:
                await self._transition(
                    job_id,
                    expected=record.status,
                    target=JobStatus.FAILED,
                    error=_error_payload(exc),
                )
        finally:
            try:
                await self._release_reservation(job_id)
            finally:
                self._tokens.pop(job_id, None)
                self._active.pop(job_id, None)
                self._wake.set()

    async def cancel(self, job_id: str, *, reason: str = "user request") -> bool:
        if not isinstance(reason, str) or not reason or reason != reason.strip():
            raise ValueError("reason must be a non-empty trimmed string")
        async with self._control_lock:
            record = await self.repository.get(job_id)
            if record is None or record.status.terminal:
                return False
            if record.status is JobStatus.QUEUED:
                self.queues.remove(job_id)
                await self._transition(
                    job_id,
                    expected=JobStatus.QUEUED,
                    target=JobStatus.CANCELLED,
                    error={"reason": reason},
                )
                return True
            token = self._tokens.get(job_id)
            if token is None:
                return False
            changed = token.cancel(reason)
        cancel = getattr(self.dispatcher, "cancel", None)
        if cancel is not None:
            result = cancel(job_id)
            if inspect.isawaitable(result):
                await result
        self._wake.set()
        return changed

    async def retry(self, job_id: str) -> JobRecord:
        async with self._control_lock:
            current = await self.repository.get(job_id)
            if current is not None:
                queue = self._queue(current.queue)
                try:
                    self.queues.ensure_capacity(queue)
                except QueueFullError as exc:
                    raise SchedulerBackpressure(exc) from exc
            retried = await self.repository.retry(job_id, now=self._now())
            self.queues.enqueue(
                QueueItem(
                    job_id=job_id,
                    queue=self._queue(retried.queue),
                    enqueued_at=retried.queued_at,
                    factors=self._factors(retried),
                )
            )
            await self.events.publish(
                "job.queued",
                {
                    "queue": retried.queue,
                    "attempt": retried.attempt,
                    "manual_retry": True,
                },
                job_id=job_id,
            )
            self._wake.set()
            return retried

    async def run_once(self) -> bool:
        active_before = set(self._active)
        launched = await self._launch_next()
        new_jobs = set(self._active) - active_before
        if new_jobs:
            await asyncio.gather(
                *(self._active[job_id] for job_id in new_jobs if job_id in self._active)
            )
        return launched

    async def run_until_idle(self) -> bool:
        while True:
            blocked = False
            while len(self._active) < self.max_concurrency and len(self.queues) > 0:
                if not await self._launch_next():
                    blocked = True
                    break
            if self._active:
                await asyncio.wait(
                    tuple(self._active.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                continue
            if len(self.queues) == 0:
                return True
            if blocked:
                return False

    async def run(self) -> None:
        self._stop_requested = False
        while not self._stop_requested:
            self._wake.clear()
            idle = await self.run_until_idle()
            if self._stop_requested:
                break
            if not idle:
                await self._sleeper(self.resource_poll_interval)
                continue
            await self._wake.wait()

    def stop(self) -> None:
        self._stop_requested = True
        self._wake.set()

    @property
    def pending_count(self) -> int:
        return len(self.queues)

    @property
    def active_count(self) -> int:
        return len(self._active)


__all__ = [
    "AdmissionBroker",
    "DispatchResult",
    "ResourceAdmissionFailure",
    "Scheduler",
    "SchedulerBackpressure",
    "WorkerDispatcher",
]
