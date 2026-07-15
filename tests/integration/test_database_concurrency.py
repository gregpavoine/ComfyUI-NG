from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

from comfyng.config.models import DatabaseSettings


def database_settings(path: Path) -> DatabaseSettings:
    return DatabaseSettings(path=path, busy_timeout_ms=5_000)


async def open_database(path: Path):
    from comfyng.database import Database

    database = Database(database_settings(path))
    await database.open()
    return database


def _write_setting_in_process(path: str, key: str) -> tuple[str, int, int, int]:
    async def scenario() -> tuple[str, int, int, int]:
        from comfyng.database import Repositories

        database = await open_database(Path(path))
        repositories = Repositories(database)
        await repositories.settings.create({"key": key, "value_json": "true"})
        async with database.connection() as connection:
            journal = (await connection.execute_fetchall("PRAGMA journal_mode"))[0][0]
            foreign_keys = (
                await connection.execute_fetchall("PRAGMA foreign_keys")
            )[0][0]
            timeout = (await connection.execute_fetchall("PRAGMA busy_timeout"))[0][0]
            version = (await connection.execute_fetchall("PRAGMA user_version"))[0][0]
        return str(journal), foreign_keys, timeout, version

    return asyncio.run(scenario())


def test_wal_allows_a_writer_while_a_reader_holds_a_snapshot(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import Repositories

        database = await open_database(tmp_path / "state.db")
        repositories = Repositories(database)
        await repositories.settings.create({"key": "before", "value_json": "true"})

        async with database.connection() as reader:
            await reader.execute("BEGIN")
            initial = await reader.execute_fetchall("SELECT key FROM settings ORDER BY key")

            await asyncio.wait_for(
                repositories.settings.create(
                    {"key": "during-read", "value_json": "true"}
                ),
                timeout=2,
            )
            snapshot = await reader.execute_fetchall("SELECT key FROM settings ORDER BY key")
            await reader.rollback()

        async with database.connection() as reader:
            current = await reader.execute_fetchall("SELECT key FROM settings ORDER BY key")

        assert [row[0] for row in initial] == ["before"]
        assert [row[0] for row in snapshot] == ["before"]
        assert [row[0] for row in current] == ["before", "during-read"]

    asyncio.run(scenario())


def test_concurrent_version_allocations_have_no_duplicates(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.database import Repositories

        database = await open_database(tmp_path / "state.db")
        repositories = Repositories(database)
        await repositories.workflows.create({"id": "wf-1", "name": "Concurrent"})

        workflow_rows = await asyncio.gather(
            *(
                repositories.workflow_versions.create_version(
                    "wf-1", graph_json=f'{{"task": {task}}}'
                )
                for task in range(24)
            )
        )
        artifact_rows = await asyncio.gather(
            *(
                repositories.artifacts.create_version(
                    owner_type="workflow",
                    owner_id="wf-1",
                    name="preview",
                    kind="image/png",
                    uri=f"cas://preview-{task}",
                )
                for task in range(24)
            )
        )

        assert sorted(row["version"] for row in workflow_rows) == list(range(1, 25))
        assert sorted(row["version"] for row in artifact_rows) == list(range(1, 25))

    asyncio.run(scenario())


def test_multiple_processes_migrate_and_write_the_same_database(tmp_path: Path) -> None:
    database_path = tmp_path / "shared.db"
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        futures = [
            executor.submit(_write_setting_in_process, str(database_path), f"process-{index}")
            for index in range(8)
        ]
        pragmas = [future.result(timeout=30) for future in futures]

    async def verify() -> None:
        database = await open_database(database_path)
        async with database.connection() as connection:
            rows = await connection.execute_fetchall(
                "SELECT key FROM settings WHERE key LIKE 'process-%' ORDER BY key"
            )
        assert [row[0] for row in rows] == [f"process-{index}" for index in range(8)]

    assert pragmas == [("wal", 1, 5_000, 1)] * 8
    asyncio.run(verify())
