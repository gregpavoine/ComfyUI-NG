from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import aiosqlite

from comfyng.config.models import DatabaseSettings

from .migrations import migrate as apply_migrations


TransactionMode = Literal["DEFERRED", "IMMEDIATE", "EXCLUSIVE"]


class Database:
    """Connection factory for one multi-process-safe SQLite database."""

    def __init__(
        self,
        settings: DatabaseSettings | Path,
        *,
        busy_timeout_ms: int | None = None,
    ) -> None:
        if isinstance(settings, DatabaseSettings):
            self.path = settings.path
            self.busy_timeout_ms = settings.busy_timeout_ms
        else:
            self.path = Path(settings)
            self.busy_timeout_ms = 5_000 if busy_timeout_ms is None else busy_timeout_ms
        if self.busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self._opened = False
        self._open_lock = asyncio.Lock()

    async def _connect(self) -> aiosqlite.Connection:
        connection = await aiosqlite.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA synchronous = NORMAL")
            return connection
        except BaseException:
            await connection.close()
            raise

    async def open(self) -> Database:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self._open_lock:
            if not self._opened:
                await self.migrate()
                self._opened = True
        return self

    async def close(self) -> None:
        self._opened = False

    async def migrate(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = await self._connect()
        try:
            return await apply_migrations(connection)
        finally:
            await connection.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        if not self._opened:
            await self.open()
        connection = await self._connect()
        try:
            yield connection
        finally:
            await connection.close()

    @asynccontextmanager
    async def transaction(
        self,
        mode: TransactionMode = "IMMEDIATE",
    ) -> AsyncIterator[aiosqlite.Connection]:
        if mode not in {"DEFERRED", "IMMEDIATE", "EXCLUSIVE"}:
            raise ValueError(f"unsupported transaction mode: {mode}")
        async with self.connection() as connection:
            await connection.execute(f"BEGIN {mode}")
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def __aenter__(self) -> Database:
        return await self.open()

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def repositories(self):
        from .repositories import Repositories

        return Repositories(self)
