from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode",
    ("reference", "index", "move", "copy", "symlink", "hardlink"),
)
def test_import_external_supports_all_six_declared_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    from comfyng.storage.cas import CAS
    from comfyng.storage.imports import ImportMode

    source = tmp_path / "external" / f"source-{mode}.weights"
    source.parent.mkdir()
    payload = f"payload:{mode}".encode()
    source.write_bytes(payload)
    source_inode = source.stat().st_ino
    digest = hashlib.sha256(payload).hexdigest()
    cas = CAS(tmp_path / "storage")

    result = cas.import_external(
        source,
        mode=mode,
        logical_path=f"models/example/{mode}.weights",
        expected_sha256=digest,
    )

    assert result.mode is ImportMode(mode)
    assert result.digest == digest
    assert result.size_bytes == len(payload)
    assert result.source_path == source.resolve(strict=False)
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "comfyng.external-import/v1"
    assert manifest["mode"] == mode
    assert manifest["sha256"] == digest
    assert manifest["size_bytes"] == len(payload)

    if mode == "move":
        assert not source.exists()
    else:
        assert source.read_bytes() == payload

    if mode in {"copy", "move"}:
        assert result.blob is not None
        assert result.blob.path.read_bytes() == payload
        assert result.storage_path.read_bytes() == payload
    else:
        assert result.blob is None

    if mode == "symlink":
        assert result.storage_path.is_symlink()
        assert result.storage_path.resolve() == source.resolve()
    elif mode == "hardlink":
        assert result.storage_path.stat().st_ino == source_inode
    elif mode in {"reference", "index"}:
        assert result.storage_path == source.resolve()


def test_import_manifest_is_idempotent_and_immutable(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS, StorageConflict

    source = tmp_path / "external.weights"
    source.write_bytes(b"version-one")
    cas = CAS(tmp_path / "storage")

    first = cas.import_external(
        source,
        mode="index",
        logical_path="models/demo/indexed.weights",
    )
    same = cas.import_external(
        source,
        mode="index",
        logical_path="models/demo/indexed.weights",
    )
    assert same == first

    source.write_bytes(b"version-two")
    with pytest.raises(StorageConflict, match="immutable manifest"):
        cas.import_external(
            source,
            mode="index",
            logical_path="models/demo/indexed.weights",
        )

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(b"version-one").hexdigest()


def test_external_import_rejects_directory_sources_and_bad_logical_paths(
    tmp_path: Path,
) -> None:
    from comfyng.storage.cas import CAS, UnsafeStoragePath

    cas = CAS(tmp_path / "storage")

    with pytest.raises(ValueError, match="regular file"):
        cas.import_external(
            tmp_path,
            mode="copy",
            logical_path="models/not-a-file",
        )

    source = tmp_path / "weights"
    source.write_bytes(b"payload")
    with pytest.raises(UnsafeStoragePath):
        cas.import_external(
            source,
            mode="copy",
            logical_path="../outside",
        )

    assert not (tmp_path / "outside").exists()


def test_hardlink_and_symlink_imports_never_overwrite_existing_paths(
    tmp_path: Path,
) -> None:
    from comfyng.storage.cas import CAS, StorageConflict

    cas = CAS(tmp_path / "storage")
    first = tmp_path / "first.weights"
    second = tmp_path / "second.weights"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    target = "models/demo/shared.weights"
    cas.import_external(first, mode="hardlink", logical_path=target)

    with pytest.raises(StorageConflict, match="already exists"):
        cas.import_external(second, mode="symlink", logical_path=target)

    assert os.stat(cas.refs_path / target).st_ino == first.stat().st_ino
