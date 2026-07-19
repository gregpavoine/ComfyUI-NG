from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import json
import math
import time
from typing import Any, Protocol

from comfyng.core.json_values import freeze_json_value
from comfyng.database.connection import Database

from .models import EventEnvelope


class EventJournal(Protocol):
    async def append(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        job_id: str | None = None,
    ) -> EventEnvelope: ...

    async def replay(
        self,
        *,
        after_sequence: int = 0,
        job_id: str | None = None,
        event_types: tuple[str, ...] = (),
        limit: int = 10_000,
    ) -> tuple[EventEnvelope, ...]: ...


class InMemoryEventJournal:
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._events: list[EventEnvelope] = []
        self._stream_sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("event clock must return a finite non-negative number")
        return float(value)

    async def append(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        job_id: str | None = None,
    ) -> EventEnvelope:
        stream = job_id or "__system__"
        async with self._lock:
            stream_sequence = self._stream_sequences.get(stream, 0) + 1
            event = EventEnvelope(
                sequence=len(self._events) + 1,
                stream_sequence=stream_sequence,
                event_type=event_type,
                emitted_at=self._now(),
                job_id=job_id,
                payload={} if payload is None else payload,
            )
            self._events.append(event)
            self._stream_sequences[stream] = stream_sequence
            return event

    async def replay(
        self,
        *,
        after_sequence: int = 0,
        job_id: str | None = None,
        event_types: tuple[str, ...] = (),
        limit: int = 10_000,
    ) -> tuple[EventEnvelope, ...]:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not isinstance(event_types, tuple) or not all(
            isinstance(item, str) and item for item in event_types
        ):
            raise ValueError("event_types must be a tuple of strings")
        selected_types = frozenset(event_types)
        async with self._lock:
            result = tuple(
                event
                for event in self._events
                if event.sequence > after_sequence
                and (job_id is None or event.job_id == job_id)
                and (not selected_types or event.event_type in selected_types)
            )
        return result[:limit]

    async def latest_sequence(self) -> int:
        async with self._lock:
            return len(self._events)


def _timestamp(value: str) -> float:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _row_event(row: Mapping[str, Any]) -> EventEnvelope:
    payload = json.loads(row["payload_json"])
    return EventEnvelope(
        sequence=int(row["id"]),
        stream_sequence=int(row["sequence"]),
        event_type=str(row["event_type"]),
        emitted_at=_timestamp(str(row["created_at"])),
        job_id=str(row["job_id"]),
        payload=payload,
    )


class SqliteEventJournal:
    """Durable job-event journal backed by the V1 SQLite schema."""

    def __init__(self, database: Database) -> None:
        if not isinstance(database, Database):
            raise ValueError("database must be Database")
        self.database = database

    async def append(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        job_id: str | None = None,
    ) -> EventEnvelope:
        if job_id is None:
            raise ValueError("the V1 SQLite job journal requires job_id")
        # Validate and deep-copy before entering the write transaction.
        validated = EventEnvelope(
            sequence=1,
            stream_sequence=1,
            event_type=event_type,
            emitted_at=0,
            job_id=job_id,
            payload={} if payload is None else payload,
        )
        frozen = freeze_json_value(validated.payload, path="$.payload")
        encoded = json.dumps(
            frozen,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        async with self.database.transaction("IMMEDIATE") as connection:
            row = await (
                await connection.execute(
                    "INSERT INTO job_events "
                    "(job_id, sequence, event_type, payload_json) "
                    "SELECT ?, COALESCE(MAX(sequence), 0) + 1, ?, ? "
                    "FROM job_events WHERE job_id = ? RETURNING *",
                    (job_id, event_type, encoded, job_id),
                )
            ).fetchone()
        if row is None:
            raise RuntimeError("job event insert returned no row")
        return _row_event(dict(row))

    async def replay(
        self,
        *,
        after_sequence: int = 0,
        job_id: str | None = None,
        event_types: tuple[str, ...] = (),
        limit: int = 10_000,
    ) -> tuple[EventEnvelope, ...]:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not isinstance(event_types, tuple) or not all(
            isinstance(item, str) and item for item in event_types
        ):
            raise ValueError("event_types must be a tuple of strings")
        clauses = ["id > ?"]
        parameters: list[Any] = [after_sequence]
        if job_id is not None:
            clauses.append("job_id = ?")
            parameters.append(job_id)
        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            parameters.extend(event_types)
        parameters.append(limit)
        async with self.database.connection() as connection:
            rows = await connection.execute_fetchall(
                "SELECT * FROM job_events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY id ASC LIMIT ?",
                tuple(parameters),
            )
        return tuple(_row_event(dict(row)) for row in rows)

    async def latest_sequence(self) -> int:
        async with self.database.connection() as connection:
            row = await (
                await connection.execute("SELECT COALESCE(MAX(id), 0) FROM job_events")
            ).fetchone()
        return 0 if row is None else int(row[0])


__all__ = ["EventJournal", "InMemoryEventJournal", "SqliteEventJournal"]
