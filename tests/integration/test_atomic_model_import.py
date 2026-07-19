from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

import pytest


def _shard(path: Path, tensor: str, value: bytes = b"12345678") -> Path:
    header = {
        tensor: {
            "dtype": "F16",
            "shape": [2, 2],
            "data_offsets": [0, len(value)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + value)
    return path


async def _registry(tmp_path: Path):
    from comfyng.database import Database, Repositories
    from comfyng.models.registry import ModelRegistry
    from comfyng.storage.cas import CAS

    database = await Database(tmp_path / "state.db").open()
    cas = CAS(tmp_path / "storage")
    return database, cas, ModelRegistry(cas, Repositories(database))


def test_multifile_model_becomes_available_with_every_file_and_manifest(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from comfyng.models.registry import ModelFile

        database, cas, registry = await _registry(tmp_path)
        first = _shard(tmp_path / "renamed-sd15-a", "transformer.double_blocks.0.a")
        second = _shard(tmp_path / "renamed-sdxl-b", "transformer.double_blocks.0.b")

        registered = await registry.import_model(
            "Atomic Flux",
            [
                ModelFile(first, logical_name="transformer/z-shard.safetensors"),
                ModelFile(second, logical_name="transformer/a-shard.safetensors"),
            ],
            config={"model_type": "flux"},
        )

        model_row = await database.repositories.models.get(registered.handle.id.hex)
        file_rows = await database.repositories.model_files.list(
            filters={"model_id": registered.handle.id.hex}
        )
        manifest = json.loads(registered.manifest_path.read_text(encoding="utf-8"))
        assert model_row is not None and model_row["status"] == "available"
        assert len(file_rows) == len(registered.files) == len(manifest["files"]) == 2
        assert {row["sha256"] for row in file_rows} == {
            item.digest for item in registered.files
        }
        assert cas.references_for(registered.files[0].digest) == frozenset(
            {f"model-{registered.handle.id.hex}"}
        )
        loaded = await registry.get(registered.handle.id)
        assert loaded is not None
        assert loaded.files == registered.files
        await database.close()

    asyncio.run(scenario())


def test_invalid_second_shard_publishes_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.models.inspection import ModelInspectionError

        database, cas, registry = await _registry(tmp_path)
        first = _shard(tmp_path / "valid", "transformer.double_blocks.0.a")
        invalid = tmp_path / "invalid"
        invalid.write_bytes(b"truncated")

        with pytest.raises(ModelInspectionError):
            await registry.import_model(
                "Never Visible",
                [first, invalid],
                config={"model_type": "flux"},
            )

        assert await database.repositories.models.list() == []
        assert await database.repositories.model_files.list() == []
        assert list(cas.blobs_path.iterdir()) == []
        assert list((cas.manifests_path / "models").glob("*.json")) == []
        await database.close()

    asyncio.run(scenario())


def test_failed_move_import_restores_source_and_rolls_back_publication(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, cas, registry = await _registry(tmp_path)
        first = _shard(tmp_path / "first", "transformer.double_blocks.0.first")
        await registry.import_model(
            "Duplicate Name", [first], config={"model_type": "flux"}
        )
        moved = _shard(tmp_path / "must-be-restored", "transformer.double_blocks.0.second")
        original = moved.read_bytes()

        with pytest.raises(sqlite3.IntegrityError):
            await registry.import_model(
                "Duplicate Name",
                [moved],
                mode="move",
                config={"model_type": "flux"},
            )

        assert moved.read_bytes() == original
        models = await database.repositories.models.list()
        assert len(models) == 1 and models[0]["status"] == "available"
        assert len(cas.iter_references()) == 1
        assert len(list((cas.manifests_path / "models").glob("*.json"))) == 1
        await database.close()

    asyncio.run(scenario())


def test_concurrent_model_imports_publish_complete_independent_records(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, cas, registry = await _registry(tmp_path)
        paths = [
            _shard(
                tmp_path / f"model-{index}",
                f"transformer.double_blocks.0.model_{index}",
                value=index.to_bytes(8, "big"),
            )
            for index in range(12)
        ]

        registered = await asyncio.gather(
            *(
                registry.import_model(
                    f"Concurrent {index}",
                    [path],
                    config={"model_type": "flux"},
                )
                for index, path in enumerate(paths)
            )
        )

        rows = await database.repositories.models.list(limit=20)
        assert len(rows) == len(registered) == 12
        assert {row["status"] for row in rows} == {"available"}
        assert len(cas.iter_references()) == 12
        assert len(list((cas.manifests_path / "models").glob("*.json"))) == 12
        await database.close()

    asyncio.run(scenario())
