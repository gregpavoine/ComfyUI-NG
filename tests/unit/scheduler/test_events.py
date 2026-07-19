from __future__ import annotations

import asyncio
from pathlib import Path

from comfyng.events.bus import EventBus, EventFilter
from comfyng.events.journal import InMemoryEventJournal
from comfyng.events.models import EventEnvelope


class Clock:
    value = 10.0

    def __call__(self) -> float:
        self.value += 1
        return self.value


def test_journal_is_replayable_with_global_and_stream_sequences() -> None:
    async def scenario() -> None:
        journal = InMemoryEventJournal(clock=Clock())
        first = await journal.append("job.created", {"x": 1}, job_id="job-a")
        second = await journal.append("job.queued", {}, job_id="job-a")
        third = await journal.append("job.created", {}, job_id="job-b")

        assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
        assert (first.stream_sequence, second.stream_sequence) == (1, 2)
        assert third.stream_sequence == 1
        assert await journal.replay(after_sequence=1) == (second, third)
        assert await journal.replay(after_sequence=0, job_id="job-a") == (
            first,
            second,
        )

        encoded = second.to_json()
        assert EventEnvelope.from_json(encoded) == second

    asyncio.run(scenario())


def test_sqlite_journal_survives_reconstruction(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import Database
        from comfyng.events.journal import SqliteEventJournal

        database = Database(tmp_path / "events.db")
        await database.open()
        repositories = database.repositories
        await repositories.workflows.create({"id": "wf", "name": "Workflow"})
        version = await repositories.workflow_versions.create_version(
            "wf",
            graph_json={},
        )
        await repositories.jobs.create(
            {
                "id": "durable-job",
                "workflow_id": "wf",
                "workflow_version_id": version["id"],
            }
        )

        first_journal = SqliteEventJournal(database)
        first = await first_journal.append(
            "job.created",
            {"durable": True},
            job_id="durable-job",
        )
        second = await first_journal.append(
            "job.queued",
            {},
            job_id="durable-job",
        )

        reconstructed = SqliteEventJournal(database)
        assert await reconstructed.replay(after_sequence=first.sequence) == (second,)
        assert (first.stream_sequence, second.stream_sequence) == (1, 2)

    asyncio.run(scenario())


def test_bus_persists_before_filtered_delivery_and_can_replay_after_reconnect() -> None:
    async def scenario() -> None:
        journal = InMemoryEventJournal(clock=Clock())
        bus = EventBus(journal)
        subscription = bus.subscribe(
            EventFilter(job_id="job-a", event_types=("job.running",))
        )

        await bus.publish("job.created", {}, job_id="job-a")
        running = await bus.publish("job.running", {"step": 1}, job_id="job-a")
        assert await subscription.get() == running
        await subscription.close()

        reconnected = EventBus(journal)
        assert await reconnected.replay(after_sequence=1) == (running,)

    asyncio.run(scenario())
