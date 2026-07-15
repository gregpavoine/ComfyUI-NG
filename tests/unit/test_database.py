from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3

import pytest

from comfyng.config.models import DatabaseSettings


APPLICATION_TABLES = {
    "models",
    "model_files",
    "model_sources",
    "loras",
    "plugins",
    "plugin_versions",
    "node_types",
    "workflows",
    "workflow_versions",
    "jobs",
    "job_events",
    "artifacts",
    "workers",
    "benchmarks",
    "provider_accounts",
    "downloads",
    "cache_entries",
    "settings",
}


def database_settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(path=path, busy_timeout_ms=5_000)


async def open_database(path: Path):
    from comfyng.database import Database

    database = Database(database_settings(path))
    assert await database.open() is database
    return database


def test_open_applies_connection_pragmas_and_exact_schema(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import APPLICATION_TABLES as declared_tables

        database = await open_database(tmp_path / "nested" / "state.db")

        assert set(declared_tables) == APPLICATION_TABLES
        for _ in range(2):
            async with database.connection() as connection:
                journal_mode = await connection.execute_fetchall("PRAGMA journal_mode")
                foreign_keys = await connection.execute_fetchall("PRAGMA foreign_keys")
                busy_timeout = await connection.execute_fetchall("PRAGMA busy_timeout")
                user_version = await connection.execute_fetchall("PRAGMA user_version")
                rows = await connection.execute_fetchall(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )

            assert journal_mode[0][0].lower() == "wal"
            assert foreign_keys[0][0] == 1
            assert busy_timeout[0][0] == 5_000
            assert user_version[0][0] == 1
            assert {row[0] for row in rows} == APPLICATION_TABLES

    asyncio.run(scenario())


def test_migration_is_idempotent_and_foreign_keys_are_enforced(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = await open_database(tmp_path / "state.db")
        await database.migrate()
        await database.open()

        async with database.connection() as connection:
            before = await connection.execute_fetchall(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'index') ORDER BY name"
            )

        await database.migrate()

        async with database.connection() as connection:
            after = await connection.execute_fetchall(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'index') ORDER BY name"
            )
            with pytest.raises(sqlite3.IntegrityError):
                await connection.execute(
                    "INSERT INTO model_files "
                    "(id, model_id, kind, path, sha256, size_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("file-1", "missing", "weights", "/tmp/model", "a" * 64, 1),
                )

        assert [tuple(row) for row in after] == [tuple(row) for row in before]

    asyncio.run(scenario())


def test_transaction_commits_and_rolls_back(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = await open_database(tmp_path / "state.db")

        with pytest.raises(RuntimeError, match="rollback"):
            async with database.transaction() as connection:
                await connection.execute(
                    "INSERT INTO settings (key, value_json) VALUES (?, ?)",
                    ("rolled-back", "true"),
                )
                raise RuntimeError("rollback")

        async with database.connection() as connection:
            missing = await connection.execute_fetchall(
                "SELECT key FROM settings WHERE key = ?", ("rolled-back",)
            )
        assert missing == []

        async with database.transaction() as connection:
            await connection.execute(
                "INSERT INTO settings (key, value_json) VALUES (?, ?)",
                ("committed", "true"),
            )

        async with database.connection() as connection:
            committed = await connection.execute_fetchall(
                "SELECT key FROM settings WHERE key = ?", ("committed",)
            )
        assert committed[0][0] == "committed"

    asyncio.run(scenario())


def test_repositories_expose_all_tables_and_transactional_crud(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import Repositories, TableRepository

        database = await open_database(tmp_path / "state.db")
        repositories = Repositories(database)

        assert set(repositories.table_names) == APPLICATION_TABLES
        for table in APPLICATION_TABLES:
            repository = repositories.for_table(table)
            assert isinstance(repository, TableRepository)
            assert repository is getattr(repositories, table)

        created = await repositories.settings.create(
            {"key": "theme", "value_json": '"dark"'}
        )
        assert created["key"] == "theme"
        assert await repositories.settings.get("theme") == created
        assert [row["key"] for row in await repositories.settings.list()] == ["theme"]

        changed = await repositories.settings.update(
            "theme",
            {"value_json": '"light"'},
            expected={"value_json": '"dark"'},
        )
        rejected = await repositories.settings.update(
            "theme",
            {"value_json": '"system"'},
            expected={"value_json": '"dark"'},
        )
        assert changed is not None and changed["value_json"] == '"light"'
        assert rejected is None

        with pytest.raises(RuntimeError, match="abort"):
            async with repositories.transaction() as transaction:
                await transaction.settings.create(
                    {"key": "temporary", "value_json": "null"}
                )
                raise RuntimeError("abort")
        assert await repositories.settings.get("temporary") is None

        assert await repositories.settings.delete("theme") is True
        assert await repositories.settings.delete("theme") is False

    asyncio.run(scenario())


def test_workflow_and_artifact_versions_are_monotonic(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import Repositories

        database = await open_database(tmp_path / "state.db")
        repositories = Repositories(database)
        await repositories.workflows.create({"id": "wf-1", "name": "Portrait"})

        workflow_versions = [
            await repositories.workflow_versions.create_version(
                "wf-1", graph_json=f'{{"revision": {revision}}}'
            )
            for revision in range(3)
        ]
        artifact_versions = [
            await repositories.artifacts.create_version(
                owner_type="workflow",
                owner_id="wf-1",
                name="preview",
                kind="image/png",
                uri=f"cas://preview-{revision}",
            )
            for revision in range(3)
        ]

        assert [row["version"] for row in workflow_versions] == [1, 2, 3]
        assert [row["version"] for row in artifact_versions] == [1, 2, 3]
        workflow = await repositories.workflows.get("wf-1")
        assert workflow is not None and workflow["current_version"] == 3

    asyncio.run(scenario())


def test_state_transitions_and_idempotency_are_atomic(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import IdempotencyConflict, Repositories

        database = await open_database(tmp_path / "state.db")
        repositories = Repositories(database)
        await repositories.workflows.create({"id": "wf-1", "name": "Portrait"})
        version = await repositories.workflow_versions.create_version(
            "wf-1", graph_json="{}"
        )
        values = {
            "id": "job-1",
            "workflow_id": "wf-1",
            "workflow_version_id": version["id"],
            "status": "queued",
            "inputs_json": "{}",
            "execution_json": "{}",
            "priority": 80,
        }

        first, was_created = await repositories.jobs.create_idempotent(
            "request-1", "hash-a", values
        )
        replay, replay_created = await repositories.jobs.create_idempotent(
            "request-1", "hash-a", {**values, "id": "job-ignored"}
        )

        assert was_created is True
        assert replay_created is False
        assert replay["id"] == first["id"]
        with pytest.raises(IdempotencyConflict):
            await repositories.jobs.create_idempotent(
                "request-1", "hash-b", {**values, "id": "job-conflict"}
            )

        results = await asyncio.gather(
            repositories.jobs.transition_state(
                "job-1", expected=("queued",), target="running"
            ),
            repositories.jobs.transition_state(
                "job-1", expected=("queued",), target="cancelled"
            ),
        )
        final = await repositories.jobs.get("job-1")

        assert sorted(results) == [False, True]
        assert final is not None and final["status"] in {"running", "cancelled"}

    asyncio.run(scenario())
