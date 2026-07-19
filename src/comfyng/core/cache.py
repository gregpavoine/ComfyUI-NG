from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import time
from typing import Any, Protocol

from .json_values import freeze_json_value


def _key(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
    ):
        raise ValueError("cache key must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("cache key must contain valid Unicode") from exc
    return value


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed_at: float
    expires_at: float | None = None
    hit_count: int = 0

    def __post_init__(self) -> None:
        _key(self.key)
        for name in ("size_bytes", "hit_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("created_at", "last_accessed_at"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.expires_at is not None and (
            not math.isfinite(self.expires_at) or self.expires_at < self.created_at
        ):
            raise ValueError("expires_at must be finite and >= created_at")
        if self.last_accessed_at < self.created_at:
            raise ValueError("last_accessed_at cannot precede created_at")
        object.__setattr__(self, "value", freeze_json_value(self.value, path="$.value"))


class NodeResultCache(Protocol):
    async def get(self, key: str) -> CacheEntry | None: ...
    async def put(
        self,
        key: str,
        value: Any,
        *,
        size_bytes: int = 0,
        ttl_seconds: float | None = None,
    ) -> CacheEntry: ...


@dataclass(slots=True)
class _Stored:
    entry: CacheEntry
    access_order: int


class CacheCapacityError(RuntimeError):
    pass


class InMemoryNodeResultCache:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = 10_000,
        max_bytes: int = 1024**3,
    ) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise ValueError("max_bytes must be a non-negative integer")
        self._clock = clock
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: dict[str, _Stored] = {}
        self._bytes = 0
        self._order = 0
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("cache clock must return a finite non-negative number")
        return float(value)

    def _next_order(self) -> int:
        self._order += 1
        return self._order

    def _expired(self, entry: CacheEntry, now: float) -> bool:
        return entry.expires_at is not None and now >= entry.expires_at

    def _purge_expired(self, now: float) -> None:
        expired = tuple(
            key
            for key, stored in self._entries.items()
            if self._expired(stored.entry, now)
        )
        for key in expired:
            self._bytes -= self._entries[key].entry.size_bytes
            del self._entries[key]

    async def get(self, key: str) -> CacheEntry | None:
        resolved = _key(key)
        async with self._lock:
            stored = self._entries.get(resolved)
            if stored is None:
                return None
            now = self._now()
            if self._expired(stored.entry, now):
                self._bytes -= stored.entry.size_bytes
                del self._entries[resolved]
                return None
            stored.entry = replace(
                stored.entry,
                last_accessed_at=now,
                hit_count=stored.entry.hit_count + 1,
            )
            stored.access_order = self._next_order()
            return stored.entry

    async def put(
        self,
        key: str,
        value: Any,
        *,
        size_bytes: int = 0,
        ttl_seconds: float | None = None,
    ) -> CacheEntry:
        resolved = _key(key)
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer")
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be finite and positive or None")
        if size_bytes > self.max_bytes:
            raise CacheCapacityError("entry exceeds cache byte limit")
        now = self._now()
        entry = CacheEntry(
            key=resolved,
            value=value,
            size_bytes=size_bytes,
            created_at=now,
            last_accessed_at=now,
            expires_at=None if ttl_seconds is None else now + ttl_seconds,
        )
        async with self._lock:
            self._purge_expired(now)
            previous = self._entries.pop(resolved, None)
            if previous is not None:
                self._bytes -= previous.entry.size_bytes
            self._entries[resolved] = _Stored(entry, self._next_order())
            self._bytes += size_bytes
            self._evict_to_limits()
            return entry

    def _evict_to_limits(self) -> None:
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            key, stored = min(
                self._entries.items(),
                key=lambda item: (item[1].access_order, item[0]),
            )
            self._bytes -= stored.entry.size_bytes
            del self._entries[key]

    async def delete(self, key: str) -> bool:
        resolved = _key(key)
        async with self._lock:
            stored = self._entries.pop(resolved, None)
            if stored is None:
                return False
            self._bytes -= stored.entry.size_bytes
            return True

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._bytes = 0
            return count

    async def size(self) -> int:
        async with self._lock:
            self._purge_expired(self._now())
            return len(self._entries)

    async def total_bytes(self) -> int:
        async with self._lock:
            self._purge_expired(self._now())
            return self._bytes

    async def evict_until(self, *, target_bytes: int) -> tuple[CacheEntry, ...]:
        if (
            isinstance(target_bytes, bool)
            or not isinstance(target_bytes, int)
            or target_bytes < 0
        ):
            raise ValueError("target_bytes must be a non-negative integer")
        evicted: list[CacheEntry] = []
        async with self._lock:
            while self._bytes > target_bytes and self._entries:
                key, stored = min(
                    self._entries.items(),
                    key=lambda item: (item[1].access_order, item[0]),
                )
                del self._entries[key]
                self._bytes -= stored.entry.size_bytes
                evicted.append(stored.entry)
        return tuple(evicted)


__all__ = [
    "CacheCapacityError",
    "CacheEntry",
    "InMemoryNodeResultCache",
    "NodeResultCache",
]
