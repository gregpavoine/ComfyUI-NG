from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .journal import EventJournal
from .models import EventEnvelope


@dataclass(frozen=True, slots=True)
class EventFilter:
    job_id: str | None = None
    event_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.job_id is not None and (
            not isinstance(self.job_id, str) or not self.job_id
        ):
            raise ValueError("job_id must be a non-empty string or None")
        if not isinstance(self.event_types, tuple) or not all(
            isinstance(item, str) and item for item in self.event_types
        ):
            raise ValueError("event_types must be a tuple of non-empty strings")

    def matches(self, event: EventEnvelope) -> bool:
        return (self.job_id is None or event.job_id == self.job_id) and (
            not self.event_types or event.event_type in self.event_types
        )


class EventSubscription:
    def __init__(
        self,
        bus: EventBus,
        event_filter: EventFilter,
        *,
        max_buffer: int,
    ) -> None:
        self._bus = bus
        self.filter = event_filter
        self._queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=max_buffer)
        self._closed = False
        self.dropped_events = 0

    def _deliver(self, event: EventEnvelope) -> None:
        if self._closed or not self.filter.matches(event):
            return
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped_events += 1
        self._queue.put_nowait(event)

    async def get(self) -> EventEnvelope:
        if self._closed and self._queue.empty():
            raise RuntimeError("subscription is closed")
        return await self._queue.get()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus._unsubscribe(self)

    def __aiter__(self) -> EventSubscription:
        return self

    async def __anext__(self) -> EventEnvelope:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        return await self.get()


class EventBus:
    def __init__(self, journal: EventJournal) -> None:
        self.journal = journal
        self._subscriptions: set[EventSubscription] = set()

    def subscribe(
        self,
        event_filter: EventFilter | None = None,
        *,
        max_buffer: int = 256,
    ) -> EventSubscription:
        if (
            isinstance(max_buffer, bool)
            or not isinstance(max_buffer, int)
            or max_buffer < 1
        ):
            raise ValueError("max_buffer must be a positive integer")
        subscription = EventSubscription(
            self,
            event_filter or EventFilter(),
            max_buffer=max_buffer,
        )
        self._subscriptions.add(subscription)
        return subscription

    def _unsubscribe(self, subscription: EventSubscription) -> None:
        self._subscriptions.discard(subscription)

    async def publish(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        job_id: str | None = None,
    ) -> EventEnvelope:
        event = await self.journal.append(event_type, payload, job_id=job_id)
        for subscription in tuple(self._subscriptions):
            subscription._deliver(event)
        return event

    async def replay(
        self,
        *,
        after_sequence: int = 0,
        job_id: str | None = None,
        event_types: tuple[str, ...] = (),
        limit: int = 10_000,
    ) -> tuple[EventEnvelope, ...]:
        return await self.journal.replay(
            after_sequence=after_sequence,
            job_id=job_id,
            event_types=event_types,
            limit=limit,
        )


__all__ = ["EventBus", "EventFilter", "EventSubscription"]
