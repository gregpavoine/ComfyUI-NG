from __future__ import annotations

import asyncio

from comfyng.core.cache import InMemoryNodeResultCache


class Clock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


def test_cache_reuses_results_tracks_hits_and_expires() -> None:
    async def scenario() -> None:
        clock = Clock()
        cache = InMemoryNodeResultCache(clock=clock)
        await cache.put(
            "node:a", {"value_ref": "cas://a"}, size_bytes=10, ttl_seconds=5
        )

        hit = await cache.get("node:a")
        assert hit is not None
        assert hit.value == {"value_ref": "cas://a"}
        assert hit.hit_count == 1
        clock.value = 6
        assert await cache.get("node:a") is None
        assert await cache.size() == 0

    asyncio.run(scenario())


def test_cache_evicts_lru_to_bounded_entry_and_byte_limits() -> None:
    async def scenario() -> None:
        clock = Clock()
        cache = InMemoryNodeResultCache(clock=clock, max_entries=2, max_bytes=20)
        await cache.put("a", "cas://a", size_bytes=10)
        clock.value = 1
        await cache.put("b", "cas://b", size_bytes=10)
        await cache.get("a")
        clock.value = 2
        await cache.put("c", "cas://c", size_bytes=10)

        assert await cache.get("a") is not None
        assert await cache.get("b") is None
        assert await cache.get("c") is not None
        assert await cache.total_bytes() == 20

    asyncio.run(scenario())
