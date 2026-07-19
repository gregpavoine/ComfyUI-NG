from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest


def _weights(path: Path, *, metadata: dict[str, str] | None = None) -> Path:
    header: dict[str, object] = {
        "transformer.double_blocks.0.img_attn.qkv.weight": {
            "dtype": "F16",
            "shape": [2, 2],
            "data_offsets": [0, 8],
        }
    }
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(8))
    return path


async def _registry(tmp_path: Path):
    from comfyng.database import Database, Repositories
    from comfyng.models.registry import ModelRegistry
    from comfyng.storage.cas import CAS

    database = await Database(tmp_path / "state.db").open()
    cas = CAS(tmp_path / "storage")
    return database, cas, ModelRegistry(cas, Repositories(database))


def test_registry_publishes_complete_modern_model_and_evicts_runtime_state(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, cas, registry = await _registry(tmp_path)
        path = _weights(tmp_path / "sd15-misleading-name.ckpt")

        registered = await registry.import_model(
            "Portrait Flux",
            [path],
            mode="copy",
            config={"model_type": "flux"},
            metadata={"description": "local model"},
        )

        assert registered.status == "available"
        assert registered.handle.family == "flux1"
        assert registered.handle.architecture == "flux-transformer-2d"
        assert registered.handle.local_path.read_bytes() == path.read_bytes()
        assert registered.handle.size_bytes == path.stat().st_size
        assert len(registered.files) == 1
        assert registered.files[0].digest == registered.handle.sha256
        assert cas.references_for(registered.files[0].digest) == frozenset(
            {f"model-{registered.handle.id.hex}"}
        )
        manifest = json.loads(registered.manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "comfyng.model/v1"
        assert manifest["status"] == "available"
        assert manifest["license"] == "unknown"
        assert manifest["metadata"]["description"] == "local model"

        assert await registry.evict(registered.handle.id) is True
        assert await registry.evict(registered.handle.id) is False
        row = await database.repositories.models.get(registered.handle.id.hex)
        assert row is not None and row["status"] == "evicted"
        assert registered.files[0].path.exists()
        await database.close()

    asyncio.run(scenario())


def test_registry_records_provider_source_and_structured_capabilities(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from comfyng.models.registry import ModelSource

        database, _, registry = await _registry(tmp_path)
        path = _weights(tmp_path / "remote.weights")
        source = ModelSource(
            provider="huggingface",
            source_id="black-forest-labs/FLUX.1-dev",
            revision="abc123",
            metadata={"etag": "value"},
        )

        registered = await registry.import_model(
            "Remote Flux",
            [path],
            mode="copy",
            config={"model_type": "flux"},
            repository_manifest={"license": "apache-2.0"},
            provider_declaration={"family": "qwen_image"},
            model_source=source,
        )

        assert registered.handle.source_provider == "huggingface"
        assert registered.handle.source_model_id == source.source_id
        assert registered.handle.source_revision == "abc123"
        source_rows = await database.repositories.model_sources.list(
            filters={"model_id": registered.handle.id.hex}
        )
        assert len(source_rows) == 1
        assert source_rows[0]["provider"] == "huggingface"
        model_row = await database.repositories.models.get(registered.handle.id.hex)
        assert model_row is not None
        capabilities = json.loads(model_row["capabilities_json"])
        assert capabilities["family"] == "flux1"
        assert "text-to-image" in capabilities["task_types"]
        await database.close()

    asyncio.run(scenario())


def test_registry_refuses_legacy_before_storage_or_gpu_runtime_selection(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from comfyng.models.legacy import UnsupportedModelGeneration

        database, cas, registry = await _registry(tmp_path)
        path = _weights(tmp_path / "modern-looking.bin")
        modules_before = set(sys.modules)

        with pytest.raises(UnsupportedModelGeneration):
            await registry.import_model(
                "Legacy",
                [path],
                config={"model_type": "stable-diffusion-v1-5"},
            )

        assert list(cas.blobs_path.iterdir()) == []
        assert await database.repositories.models.list() == []
        assert not any(
            name == "torch" or name.startswith("torch.")
            for name in set(sys.modules) - modules_before
        )
        await database.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("logical_name", ("../escape", "/absolute", "a/../../b"))
def test_registry_rejects_file_path_traversal_before_import(
    tmp_path: Path,
    logical_name: str,
) -> None:
    async def scenario() -> None:
        from comfyng.models.registry import ModelFile
        from comfyng.storage.cas import UnsafeStoragePath

        database, cas, registry = await _registry(tmp_path)
        path = _weights(tmp_path / "weights")

        with pytest.raises(UnsafeStoragePath):
            await registry.import_model(
                "Unsafe",
                [ModelFile(path=path, logical_name=logical_name)],
                config={"model_type": "flux"},
            )

        assert list(cas.blobs_path.iterdir()) == []
        await database.close()

    asyncio.run(scenario())


def test_registry_accepts_every_external_import_mode(tmp_path: Path) -> None:
    async def scenario() -> None:
        from comfyng.storage.imports import ImportMode

        database, cas, registry = await _registry(tmp_path)
        for mode in ImportMode:
            path = _weights(
                tmp_path / f"weights-{mode.value}",
                metadata={"test.mode": mode.value},
            )
            payload = path.read_bytes()
            registered = await registry.import_model(
                f"Model {mode.value}",
                [path],
                mode=mode,
                config={"model_type": "flux"},
            )
            assert registered.files[0].import_mode is mode
            assert registered.files[0].path.read_bytes() == payload
            assert path.exists() is (mode is not ImportMode.MOVE)
            if mode in {ImportMode.COPY, ImportMode.MOVE}:
                assert cas.references_for(registered.files[0].digest) == frozenset(
                    {f"model-{registered.handle.id.hex}"}
                )
            else:
                assert not cas.blob_path(registered.files[0].digest).exists()
        await database.close()

    asyncio.run(scenario())
