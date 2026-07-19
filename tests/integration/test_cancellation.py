from __future__ import annotations

import asyncio

import pytest

from comfyng.core.cache import InMemoryNodeResultCache
from comfyng.core.jobs import InMemoryJobRepository, JobStatus, JobSubmission
from comfyng.events.bus import EventBus
from comfyng.events.journal import InMemoryEventJournal
from comfyng.resources.broker import ResourceBroker
from comfyng.resources.budgets import ResourceEstimate
from comfyng.resources.hardware import CpuInventory, HardwareInventory, MemoryInventory
from comfyng.scheduler.cancellation import CancellationCheckpoint
from comfyng.scheduler.retry import RetryPolicy
from comfyng.scheduler.scheduler import DispatchResult, Scheduler


GIB = 1024**3


class CheckpointDispatcher:
    def __init__(self, target: CancellationCheckpoint) -> None:
        self.target = target
        self.reached = asyncio.Event()
        self.resume = asyncio.Event()

    async def dispatch(self, _job, token) -> DispatchResult:
        for checkpoint in CancellationCheckpoint:
            if checkpoint is self.target:
                self.reached.set()
                await self.resume.wait()
            token.checkpoint(checkpoint)
        return DispatchResult(value={"unexpected": True})

    async def cancel(self, _job_id: str) -> None:
        self.resume.set()


@pytest.mark.parametrize("target", tuple(CancellationCheckpoint))
def test_scheduler_cancels_at_every_sampler_checkpoint(
    target: CancellationCheckpoint,
) -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        journal = InMemoryEventJournal()
        broker = ResourceBroker(
            inventory=HardwareInventory(
                cpu=CpuInventory(8, 8, "x86_64"),
                memory=MemoryInventory(16 * GIB, 16 * GIB, 0, 0),
            ),
            reserve_cpu_cores=2,
            reserve_ram_bytes=2 * GIB,
        )
        dispatcher = CheckpointDispatcher(target)
        scheduler = Scheduler(
            repository=repository,
            events=EventBus(journal),
            cache=InMemoryNodeResultCache(),
            broker=broker,
            dispatcher=dispatcher,
            retry_policy=RetryPolicy(max_attempts=1),
            max_concurrency=1,
        )
        await scheduler.submit(
            JobSubmission(
                job_id=f"cancel-{target.value}",
                resource_estimate=ResourceEstimate(cpu_cores=1, ram_bytes=GIB),
            )
        )
        runner = asyncio.create_task(scheduler.run_until_idle())
        await asyncio.wait_for(dispatcher.reached.wait(), timeout=2)
        assert await scheduler.cancel(f"cancel-{target.value}", reason="user")
        dispatcher.resume.set()
        await asyncio.wait_for(runner, timeout=2)

        record = await repository.get(f"cancel-{target.value}")
        assert record.status is JobStatus.CANCELLED
        assert broker.active_reservations == ()
        events = await journal.replay(job_id=record.job_id)
        terminal = tuple(
            item
            for item in events
            if item.event_type.startswith("job.")
            and item.event_type
            in {
                "job.completed",
                "job.failed",
                "job.cancelled",
            }
        )
        assert len(terminal) == 1
        assert terminal[0].event_type == "job.cancelled"

    asyncio.run(scenario())
