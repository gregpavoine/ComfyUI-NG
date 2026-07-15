from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import re
import sqlite3
from typing import Protocol

from .models import MigrationError


_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{4})_[a-z0-9_]+[.]sql$")


class AsyncConnection(Protocol):
    async def execute(self, sql: str, parameters: tuple[object, ...] = ...): ...
    async def execute_fetchall(
        self, sql: str, parameters: tuple[object, ...] = ...
    ): ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


def _source_migration_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "migrations"


def load_migrations(directory: Path | None = None) -> tuple[Migration, ...]:
    if directory is not None:
        entries = tuple(directory.glob("*.sql"))
        contents = ((entry.name, entry.read_text(encoding="utf-8")) for entry in entries)
    else:
        source = _source_migration_directory()
        if source.is_dir():
            entries = tuple(source.glob("*.sql"))
            contents = ((entry.name, entry.read_text(encoding="utf-8")) for entry in entries)
        else:
            resource_dir = files("comfyng.database").joinpath("migrations")
            resources = tuple(entry for entry in resource_dir.iterdir() if entry.name.endswith(".sql"))
            contents = ((entry.name, entry.read_text(encoding="utf-8")) for entry in resources)

    migrations: list[Migration] = []
    for name, sql in contents:
        match = _MIGRATION_NAME.fullmatch(name)
        if match is None:
            raise MigrationError(f"invalid migration filename: {name}")
        migrations.append(Migration(int(match.group("version")), name, sql))
    migrations.sort(key=lambda migration: migration.version)

    versions = [migration.version for migration in migrations]
    expected = list(range(1, len(migrations) + 1))
    if versions != expected:
        raise MigrationError(
            f"migration versions must be contiguous from 1: expected {expected}, got {versions}"
        )
    if not migrations:
        raise MigrationError("no database migrations were found")
    return tuple(migrations)


def _statements(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines(keepends=True):
        buffer.append(line)
        candidate = "".join(buffer).strip()
        if candidate and sqlite3.complete_statement(candidate):
            statements.append(candidate)
            buffer.clear()
    remainder = "".join(buffer).strip()
    if remainder:
        raise MigrationError("migration ends with an incomplete SQL statement")
    return tuple(statements)


async def migrate(connection: AsyncConnection, directory: Path | None = None) -> int:
    migrations = load_migrations(directory)
    await connection.execute("BEGIN IMMEDIATE")
    try:
        current = int((await connection.execute_fetchall("PRAGMA user_version"))[0][0])
        latest = migrations[-1].version
        if current > latest:
            raise MigrationError(
                f"database schema version {current} is newer than supported version {latest}"
            )

        for migration in migrations:
            if migration.version <= current:
                continue
            for statement in _statements(migration.sql):
                await connection.execute(statement)
            await connection.execute(f"PRAGMA user_version = {migration.version}")
            current = migration.version
        await connection.commit()
        return current
    except BaseException:
        await connection.rollback()
        raise
