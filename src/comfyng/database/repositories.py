from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import aiosqlite

from .connection import Database
from .models import (
    APPLICATION_TABLES,
    IdempotencyConflict,
    TABLE_SPECS,
    TableSpec,
)


Record = dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    return value


def _record(row: aiosqlite.Row | None) -> Record | None:
    return None if row is None else dict(row)


class TableRepository:
    """Validated generic CRUD bound to a known application table."""

    def __init__(
        self,
        database: Database,
        spec: TableSpec,
        connection: aiosqlite.Connection | None = None,
    ) -> None:
        self.database = database
        self.spec = spec
        self._bound_connection = connection

    def _validate_columns(self, columns: Iterable[str]) -> tuple[str, ...]:
        result = tuple(columns)
        unknown = set(result) - self.spec.columns
        if unknown:
            raise ValueError(
                f"unknown columns for {self.spec.name}: {', '.join(sorted(unknown))}"
            )
        return result

    @asynccontextmanager
    async def _connection(self, *, write: bool) -> AsyncIterator[aiosqlite.Connection]:
        if self._bound_connection is not None:
            yield self._bound_connection
        elif write:
            async with self.database.transaction("IMMEDIATE") as connection:
                yield connection
        else:
            async with self.database.connection() as connection:
                yield connection

    async def create(self, values: Mapping[str, Any]) -> Record:
        columns = self._validate_columns(values)
        if not columns:
            raise ValueError("create values cannot be empty")
        placeholders = ", ".join("?" for _ in columns)
        sql = (
            f"INSERT INTO {self.spec.name} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) RETURNING *"
        )
        parameters = tuple(_value(values[column]) for column in columns)
        async with self._connection(write=True) as connection:
            row = await (await connection.execute(sql, parameters)).fetchone()
        result = _record(row)
        if result is None:
            raise RuntimeError(f"insert into {self.spec.name} returned no row")
        return result

    async def get(self, key: Any) -> Record | None:
        async with self._connection(write=False) as connection:
            row = await (
                await connection.execute(
                    f"SELECT * FROM {self.spec.name} WHERE {self.spec.primary_key} = ?",
                    (_value(key),),
                )
            ).fetchone()
        return _record(row)

    async def list(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Record]:
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset must be non-negative")
        filter_values = {} if filters is None else dict(filters)
        filter_columns = self._validate_columns(filter_values)
        selected_order = self.spec.primary_key if order_by is None else order_by
        self._validate_columns((selected_order,))
        clauses: list[str] = []
        parameters: list[Any] = []
        for column in filter_columns:
            value = filter_values[column]
            if value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                parameters.append(_value(value))
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        direction = "DESC" if descending else "ASC"
        sql = (
            f"SELECT * FROM {self.spec.name}{where} "
            f"ORDER BY {selected_order} {direction} LIMIT ? OFFSET ?"
        )
        parameters.extend((limit, offset))
        async with self._connection(write=False) as connection:
            rows = await connection.execute_fetchall(sql, tuple(parameters))
        return [dict(row) for row in rows]

    async def update(
        self,
        key: Any,
        values: Mapping[str, Any],
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> Record | None:
        columns = self._validate_columns(values)
        if not columns:
            raise ValueError("update values cannot be empty")
        if self.spec.primary_key in columns:
            raise ValueError("primary keys cannot be updated")
        expected_values = {} if expected is None else dict(expected)
        expected_columns = self._validate_columns(expected_values)
        clauses = [f"{self.spec.primary_key} = ?"]
        parameters = [_value(values[column]) for column in columns]
        parameters.append(_value(key))
        for column in expected_columns:
            expected_value = expected_values[column]
            if expected_value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                parameters.append(_value(expected_value))
        assignments = ", ".join(f"{column} = ?" for column in columns)
        sql = (
            f"UPDATE {self.spec.name} SET {assignments} "
            f"WHERE {' AND '.join(clauses)} RETURNING *"
        )
        async with self._connection(write=True) as connection:
            row = await (await connection.execute(sql, tuple(parameters))).fetchone()
        return _record(row)

    async def delete(
        self,
        key: Any,
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> bool:
        expected_values = {} if expected is None else dict(expected)
        expected_columns = self._validate_columns(expected_values)
        clauses = [f"{self.spec.primary_key} = ?"]
        parameters = [_value(key)]
        for column in expected_columns:
            expected_value = expected_values[column]
            if expected_value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                parameters.append(_value(expected_value))
        async with self._connection(write=True) as connection:
            cursor = await connection.execute(
                f"DELETE FROM {self.spec.name} WHERE {' AND '.join(clauses)}",
                tuple(parameters),
            )
        return cursor.rowcount == 1


class StateRepository(TableRepository):
    async def transition_state(
        self,
        key: Any,
        *,
        expected: Sequence[str],
        target: str,
        values: Mapping[str, Any] | None = None,
    ) -> bool:
        expected_states = tuple(expected)
        if not expected_states:
            raise ValueError("at least one expected state is required")
        changes = {} if values is None else dict(values)
        changes["status"] = target
        if "updated_at" in self.spec.columns and "updated_at" not in changes:
            changes["updated_at"] = _utc_now()
        columns = self._validate_columns(changes)
        assignments = ", ".join(f"{column} = ?" for column in columns)
        placeholders = ", ".join("?" for _ in expected_states)
        parameters = [_value(changes[column]) for column in columns]
        parameters.extend((_value(key), *expected_states))
        sql = (
            f"UPDATE {self.spec.name} SET {assignments} "
            f"WHERE {self.spec.primary_key} = ? AND status IN ({placeholders})"
        )
        async with self._connection(write=True) as connection:
            cursor = await connection.execute(sql, tuple(parameters))
        return cursor.rowcount == 1


class ModelsRepository(StateRepository):
    pass


class ModelFilesRepository(TableRepository):
    pass


class ModelSourcesRepository(TableRepository):
    pass


class LorasRepository(StateRepository):
    pass


class PluginsRepository(StateRepository):
    pass


class PluginVersionsRepository(StateRepository):
    pass


class NodeTypesRepository(TableRepository):
    pass


class WorkflowsRepository(TableRepository):
    pass


class WorkflowVersionsRepository(TableRepository):
    async def create_version(
        self,
        workflow_id: str,
        *,
        graph_json: Any,
        metadata_json: Any = None,
    ) -> Record:
        metadata = {} if metadata_json is None else metadata_json
        async with self._connection(write=True) as connection:
            row = await (
                await connection.execute(
                    "INSERT INTO workflow_versions "
                    "(workflow_id, version, graph_json, metadata_json) "
                    "SELECT ?, COALESCE(MAX(version), 0) + 1, ?, ? "
                    "FROM workflow_versions WHERE workflow_id = ? RETURNING *",
                    (
                        workflow_id,
                        _value(graph_json),
                        _value(metadata),
                        workflow_id,
                    ),
                )
            ).fetchone()
            result = _record(row)
            if result is None:
                raise RuntimeError("workflow version insert returned no row")
            await connection.execute(
                "UPDATE workflows SET current_version = ?, updated_at = ? WHERE id = ?",
                (result["version"], _utc_now(), workflow_id),
            )
        return result


class JobsRepository(StateRepository):
    async def create_idempotent(
        self,
        key: str,
        request_hash: str,
        values: Mapping[str, Any],
    ) -> tuple[Record, bool]:
        async with self._connection(write=True) as connection:
            existing_row = await (
                await connection.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
                )
            ).fetchone()
            existing = _record(existing_row)
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflict(key)
                return existing, False

            payload = dict(values)
            payload["idempotency_key"] = key
            payload["request_hash"] = request_hash
            repository = type(self)(self.database, self.spec, connection)
            return await repository.create(payload), True


class JobEventsRepository(TableRepository):
    async def append(
        self,
        job_id: str,
        event_type: str,
        payload_json: Any = None,
    ) -> Record:
        payload = {} if payload_json is None else payload_json
        async with self._connection(write=True) as connection:
            row = await (
                await connection.execute(
                    "INSERT INTO job_events (job_id, sequence, event_type, payload_json) "
                    "SELECT ?, COALESCE(MAX(sequence), 0) + 1, ?, ? "
                    "FROM job_events WHERE job_id = ? RETURNING *",
                    (job_id, event_type, _value(payload), job_id),
                )
            ).fetchone()
        result = _record(row)
        if result is None:
            raise RuntimeError("job event insert returned no row")
        return result


class ArtifactsRepository(TableRepository):
    async def create_version(
        self,
        *,
        owner_type: str,
        owner_id: str,
        name: str,
        kind: str,
        uri: str,
        artifact_id: str | None = None,
        job_id: str | None = None,
        workflow_id: str | None = None,
        sha256: str | None = None,
        size_bytes: int = 0,
        metadata_json: Any = None,
    ) -> Record:
        metadata = {} if metadata_json is None else metadata_json
        async with self._connection(write=True) as connection:
            row = await (
                await connection.execute(
                    "INSERT INTO artifacts "
                    "(id, owner_type, owner_id, name, version, job_id, workflow_id, kind, "
                    "uri, sha256, size_bytes, metadata_json) "
                    "SELECT ?, ?, ?, ?, COALESCE(MAX(version), 0) + 1, ?, ?, ?, ?, ?, ?, ? "
                    "FROM artifacts WHERE owner_type = ? AND owner_id = ? AND name = ? "
                    "RETURNING *",
                    (
                        artifact_id or uuid4().hex,
                        owner_type,
                        owner_id,
                        name,
                        job_id,
                        workflow_id,
                        kind,
                        uri,
                        sha256,
                        size_bytes,
                        _value(metadata),
                        owner_type,
                        owner_id,
                        name,
                    ),
                )
            ).fetchone()
        result = _record(row)
        if result is None:
            raise RuntimeError("artifact version insert returned no row")
        return result


class WorkersRepository(StateRepository):
    pass


class BenchmarksRepository(StateRepository):
    pass


class ProviderAccountsRepository(StateRepository):
    pass


class DownloadsRepository(StateRepository):
    pass


class CacheEntriesRepository(TableRepository):
    pass


class SettingsRepository(TableRepository):
    async def set(self, key: str, value_json: Any) -> Record:
        async with self._connection(write=True) as connection:
            row = await (
                await connection.execute(
                    "INSERT INTO settings (key, value_json, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "value_json = excluded.value_json, updated_at = excluded.updated_at "
                    "RETURNING *",
                    (key, _value(value_json), _utc_now()),
                )
            ).fetchone()
        result = _record(row)
        if result is None:
            raise RuntimeError("setting upsert returned no row")
        return result


_REPOSITORY_TYPES: dict[str, type[TableRepository]] = {
    "models": ModelsRepository,
    "model_files": ModelFilesRepository,
    "model_sources": ModelSourcesRepository,
    "loras": LorasRepository,
    "plugins": PluginsRepository,
    "plugin_versions": PluginVersionsRepository,
    "node_types": NodeTypesRepository,
    "workflows": WorkflowsRepository,
    "workflow_versions": WorkflowVersionsRepository,
    "jobs": JobsRepository,
    "job_events": JobEventsRepository,
    "artifacts": ArtifactsRepository,
    "workers": WorkersRepository,
    "benchmarks": BenchmarksRepository,
    "provider_accounts": ProviderAccountsRepository,
    "downloads": DownloadsRepository,
    "cache_entries": CacheEntriesRepository,
    "settings": SettingsRepository,
}


class Repositories:
    """Aggregate of every durable-state repository, optionally transaction-bound."""

    table_names = APPLICATION_TABLES

    def __init__(
        self,
        database: Database,
        connection: aiosqlite.Connection | None = None,
    ) -> None:
        self.database = database
        self._connection = connection
        self._repositories: dict[str, TableRepository] = {}
        for table in APPLICATION_TABLES:
            repository = _REPOSITORY_TYPES[table](database, TABLE_SPECS[table], connection)
            self._repositories[table] = repository
            setattr(self, table, repository)

    def for_table(self, table: str) -> TableRepository:
        try:
            return self._repositories[table]
        except KeyError as error:
            raise KeyError(f"unknown application table: {table}") from error

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Self]:
        if self._connection is not None:
            yield self
            return
        async with self.database.transaction("IMMEDIATE") as connection:
            yield type(self)(self.database, connection)
