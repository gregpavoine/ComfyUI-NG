from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyng.core.cache import InMemoryNodeResultCache
from comfyng.core.jobs import (
    InMemoryJobRepository,
    JobStatus,
    JobSubmission,
    SqliteJobRepository,
)
from comfyng.events.bus import EventBus
from comfyng.events.journal import InMemoryEventJournal
from comfyng.events.journal import SqliteEventJournal
from comfyng.resources.broker import ResourceBroker
from comfyng.resources.broker import AdmissionOutcome
from comfyng.resources.budgets import ResourceEstimate
from comfyng.resources.hardware import CpuInventory, HardwareInventory, MemoryInventory
from comfyng.scheduler.queues import QueueName
from comfyng.scheduler.retry import RetryPolicy
from comfyng.scheduler.scheduler import (
    DispatchResult,
    Scheduler,
    SchedulerBackpressure,
)


GIB = 1024**3


class Clock:
    value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class Dispatcher:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.active = 0
        self.peak_active = 0

    async def dispatch(self, job, token) -> DispatchResult:
        self.calls[job.job_id] += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0)
            token.checkpoint("before_save")
            return DispatchResult(
                value={"artifact": f"cas://{job.job_id}"},
                cacheable=True,
                size_bytes=10,
            )
        finally:
            self.active -= 1

    async def cancel(self, _job_id: str) -> None:
        return None


def resource_broker() -> ResourceBroker:
    return ResourceBroker(
        inventory=HardwareInventory(
            cpu=CpuInventory(
                physical_cores=64, logical_cores=64, architecture="x86_64"
            ),
            memory=MemoryInventory(
                total_bytes=128 * GIB,
                available_bytes=128 * GIB,
                swap_total_bytes=0,
                swap_free_bytes=0,
            ),
        ),
        reserve_cpu_cores=2,
        reserve_ram_bytes=4 * GIB,
    )


def make_scheduler(*, max_queued: int = 200, max_concurrency: int = 16):
    clock = Clock()
    repository = InMemoryJobRepository()
    journal = InMemoryEventJournal(clock=clock)
    dispatcher = Dispatcher()
    broker = resource_broker()
    scheduler = Scheduler(
        repository=repository,
        events=EventBus(journal),
        cache=InMemoryNodeResultCache(clock=clock),
        broker=broker,
        dispatcher=dispatcher,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0),
        clock=clock,
        sleeper=lambda _delay: asyncio.sleep(0),
        max_queued=max_queued,
        max_concurrency=max_concurrency,
    )
    return scheduler, repository, journal, dispatcher, broker


def test_complete_lifecycle_and_cache_reuse() -> None:
    async def scenario() -> None:
        scheduler, repository, journal, dispatcher, broker = make_scheduler()
        first = JobSubmission(
            job_id="first",
            queue=QueueName.NORMAL.value,
            user_priority=50,
            payload={"prompt": "same"},
            cache_key="graph:shared",
            resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=GIB),
        )
        second = JobSubmission(
            job_id="second",
            queue=QueueName.NORMAL.value,
            user_priority=50,
            payload={"prompt": "same"},
            cache_key="graph:shared",
            resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=GIB),
        )
        await scheduler.submit(first)
        await scheduler.run_until_idle()
        await scheduler.submit(second)
        await scheduler.run_until_idle()

        assert (await repository.get("first")).status is JobStatus.COMPLETED
        cached = await repository.get("second")
        assert cached.status is JobStatus.COMPLETED
        assert cached.result == {"artifact": "cas://first"}
        assert sum(dispatcher.calls.values()) == 1
        assert broker.active_reservations == ()
        second_events = await journal.replay(job_id="second")
        assert tuple(item.event_type for item in second_events) == (
            "job.created",
            "job.queued",
            "job.preparing",
            "job.running",
            "job.completed",
        )
        assert second_events[-1].payload["cache_hit"] is True

    asyncio.run(scenario())


def test_backpressure_rejects_before_creating_an_orphan_job() -> None:
    async def scenario() -> None:
        scheduler, repository, _journal, _dispatcher, _broker = make_scheduler(
            max_queued=1
        )
        await scheduler.submit(JobSubmission(job_id="one"))
        with pytest.raises(SchedulerBackpressure) as error:
            await scheduler.submit(JobSubmission(job_id="two"))
        assert error.value.code == "QUEUE_FULL"
        assert await repository.get("two") is None

    asyncio.run(scenario())


def test_resource_backpressure_keeps_job_queued_then_dispatches_without_leak() -> None:
    class Reservation:
        released = False

        def release(self) -> bool:
            self.released = True
            return True

    class DeferredOnceBroker:
        def __init__(self) -> None:
            self.calls = 0
            self.reservation = Reservation()

        def admit(self, _estimate):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    admitted=False,
                    outcome=AdmissionOutcome.DEFERRED,
                    violations=("cpu_cores",),
                    reservation=None,
                )
            return SimpleNamespace(
                admitted=True,
                outcome=AdmissionOutcome.ADMITTED,
                violations=(),
                reservation=self.reservation,
            )

    async def scenario() -> None:
        clock = Clock()
        repository = InMemoryJobRepository()
        journal = InMemoryEventJournal(clock=clock)
        broker = DeferredOnceBroker()
        scheduler = Scheduler(
            repository=repository,
            events=EventBus(journal),
            cache=InMemoryNodeResultCache(clock=clock),
            broker=broker,
            dispatcher=Dispatcher(),
            retry_policy=RetryPolicy(max_attempts=1),
            clock=clock,
        )
        await scheduler.submit(
            JobSubmission(
                job_id="deferred",
                resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=0),
            )
        )

        assert await scheduler.run_until_idle() is False
        queued = await repository.get("deferred")
        assert queued is not None and queued.status is JobStatus.QUEUED
        assert scheduler.pending_count == 1

        assert await scheduler.run_until_idle() is True
        completed = await repository.get("deferred")
        assert completed is not None and completed.status is JobStatus.COMPLETED
        assert broker.reservation.released
        events = await journal.replay(job_id="deferred")
        assert sum(item.event_type == "job.deferred" for item in events) == 1

    asyncio.run(scenario())


def test_one_hundred_concurrent_jobs_complete_without_starvation_or_leaks() -> None:
    async def scenario() -> None:
        scheduler, repository, journal, dispatcher, broker = make_scheduler()
        submissions = tuple(
            JobSubmission(
                job_id=f"job-{index:03d}",
                queue=tuple(QueueName)[index % 6].value,
                user_priority=index % 101,
                resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=0),
            )
            for index in range(100)
        )
        await asyncio.gather(*(scheduler.submit(item) for item in submissions))
        await scheduler.run_until_idle()

        records = await repository.list()
        assert len(records) == 100
        assert all(item.status is JobStatus.COMPLETED for item in records)
        assert all(dispatcher.calls[item.job_id] == 1 for item in records)
        assert dispatcher.peak_active > 1
        assert broker.active_reservations == ()
        events = await journal.replay()
        terminal = Counter(
            item.job_id
            for item in events
            if item.event_type in {"job.completed", "job.failed", "job.cancelled"}
        )
        assert terminal == Counter({item.job_id: 1 for item in records})

    asyncio.run(scenario())


def test_automatic_retry_is_bounded_and_releases_each_reservation() -> None:
    class TemporaryFailure(RuntimeError):
        pass

    class FlakyDispatcher(Dispatcher):
        async def dispatch(self, job, token) -> DispatchResult:
            self.calls[job.job_id] += 1
            await asyncio.sleep(0)
            if self.calls[job.job_id] < 3:
                raise TemporaryFailure("temporary")
            token.checkpoint("before_save")
            return DispatchResult(value={"ok": True})

    async def scenario() -> None:
        clock = Clock()
        repository = InMemoryJobRepository()
        journal = InMemoryEventJournal(clock=clock)
        dispatcher = FlakyDispatcher()
        broker = resource_broker()
        delays: list[float] = []

        async def sleeper(delay: float) -> None:
            delays.append(delay)
            await asyncio.sleep(0)

        scheduler = Scheduler(
            repository=repository,
            events=EventBus(journal),
            cache=InMemoryNodeResultCache(clock=clock),
            broker=broker,
            dispatcher=dispatcher,
            retry_policy=RetryPolicy(
                max_attempts=3,
                base_delay_seconds=1,
                retryable_exceptions=(TemporaryFailure,),
            ),
            sleeper=sleeper,
            clock=clock,
        )
        await scheduler.submit(
            JobSubmission(
                job_id="flaky",
                max_attempts=3,
                resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=GIB),
            )
        )
        await scheduler.run_until_idle()

        record = await repository.get("flaky")
        assert record is not None
        assert record.status is JobStatus.COMPLETED
        assert record.attempt == 2
        assert dispatcher.calls["flaky"] == 3
        assert delays == [1, 2]
        assert broker.active_reservations == ()
        events = await journal.replay(job_id="flaky")
        assert sum(item.event_type == "job.retrying" for item in events) == 2
        assert sum(item.event_type == "job.completed" for item in events) == 1
        assert all(item.event_type != "job.failed" for item in events)

    asyncio.run(scenario())


def test_manual_retry_and_long_running_scheduler_loop() -> None:
    class FailingDispatcher(Dispatcher):
        async def dispatch(self, job, token) -> DispatchResult:
            self.calls[job.job_id] += 1
            raise ValueError("permanent")

    async def scenario() -> None:
        clock = Clock()
        repository = InMemoryJobRepository()
        journal = InMemoryEventJournal(clock=clock)
        failing = FailingDispatcher()
        scheduler = Scheduler(
            repository=repository,
            events=EventBus(journal),
            cache=InMemoryNodeResultCache(clock=clock),
            broker=resource_broker(),
            dispatcher=failing,
            retry_policy=RetryPolicy(max_attempts=1),
            clock=clock,
        )
        loop = asyncio.create_task(scheduler.run())
        await scheduler.submit(JobSubmission(job_id="manual", max_attempts=2))
        for _ in range(100):
            failed = await repository.get("manual")
            if failed is not None and failed.status is JobStatus.FAILED:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("job never failed")

        scheduler.dispatcher = Dispatcher()
        retried = await scheduler.retry("manual")
        assert retried.attempt == 1
        for _ in range(100):
            completed = await repository.get("manual")
            if completed is not None and completed.status is JobStatus.COMPLETED:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("retried job never completed")
        scheduler.stop()
        await asyncio.wait_for(loop, timeout=2)

    asyncio.run(scenario())


def test_cancellation_during_retry_delay_does_not_leave_running_job() -> None:
    class TemporaryFailure(RuntimeError):
        pass

    class FailingDispatcher(Dispatcher):
        async def dispatch(self, job, token) -> DispatchResult:
            self.calls[job.job_id] += 1
            raise TemporaryFailure("retry later")

    async def scenario() -> None:
        clock = Clock()
        repository = InMemoryJobRepository()
        journal = InMemoryEventJournal(clock=clock)
        delay_started = asyncio.Event()
        resume_delay = asyncio.Event()

        async def sleeper(_delay: float) -> None:
            delay_started.set()
            await resume_delay.wait()

        broker = resource_broker()
        scheduler = Scheduler(
            repository=repository,
            events=EventBus(journal),
            cache=InMemoryNodeResultCache(clock=clock),
            broker=broker,
            dispatcher=FailingDispatcher(),
            retry_policy=RetryPolicy(
                max_attempts=2,
                base_delay_seconds=1,
                retryable_exceptions=(TemporaryFailure,),
            ),
            sleeper=sleeper,
            clock=clock,
        )
        await scheduler.submit(
            JobSubmission(
                job_id="cancel-retry",
                max_attempts=2,
                resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=GIB),
            )
        )
        runner = asyncio.create_task(scheduler.run_until_idle())
        await asyncio.wait_for(delay_started.wait(), timeout=2)
        assert await scheduler.cancel("cancel-retry", reason="user")
        resume_delay.set()
        await asyncio.wait_for(runner, timeout=2)

        record = await repository.get("cancel-retry")
        assert record is not None and record.status is JobStatus.CANCELLED
        assert broker.active_reservations == ()
        assert scheduler.pending_count == 0

    asyncio.run(scenario())


def test_scheduler_job_and_events_survive_sqlite_reconstruction(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import Database

        database = Database(tmp_path / "scheduler.db")
        await database.open()
        repositories = database.repositories
        await repositories.workflows.create({"id": "wf", "name": "Durable"})
        version = await repositories.workflow_versions.create_version(
            "wf",
            graph_json={},
        )
        clock = Clock()
        scheduler = Scheduler(
            repository=SqliteJobRepository(database),
            events=EventBus(SqliteEventJournal(database)),
            cache=InMemoryNodeResultCache(clock=clock),
            broker=resource_broker(),
            dispatcher=Dispatcher(),
            retry_policy=RetryPolicy(max_attempts=1),
            clock=clock,
        )
        await scheduler.submit(
            JobSubmission(
                job_id="durable",
                workflow_id="wf",
                workflow_version_id=version["id"],
            )
        )
        await scheduler.run_until_idle()

        reconstructed_jobs = SqliteJobRepository(database)
        reconstructed_events = SqliteEventJournal(database)
        record = await reconstructed_jobs.get("durable")
        assert record is not None
        assert record.status is JobStatus.COMPLETED
        assert record.result == {"artifact": "cas://durable"}
        assert tuple(
            item.event_type
            for item in await reconstructed_events.replay(job_id="durable")
        ) == (
            "job.created",
            "job.queued",
            "job.preparing",
            "job.running",
            "job.completed",
        )

    asyncio.run(scenario())
