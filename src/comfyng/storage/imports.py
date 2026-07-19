from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .cas import (
    CAS,
    CASBlob,
    StorageConflict,
    _fsync_directory,
    _hash_path,
)
from .locks import validate_digest


class ImportMode(str, Enum):
    REFERENCE = "reference"
    INDEX = "index"
    MOVE = "move"
    COPY = "copy"
    SYMLINK = "symlink"
    HARDLINK = "hardlink"


@dataclass(frozen=True, slots=True)
class ExternalImport:
    mode: ImportMode
    digest: str
    size_bytes: int
    source_path: Path
    storage_path: Path
    manifest_path: Path
    blob: CASBlob | None


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_immutable_json(path: Path, payload: Any) -> Path:
    """Publish canonical JSON atomically, accepting only byte-identical retries."""

    path = Path(path)
    contents = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == contents:
            return path
        raise StorageConflict(f"immutable manifest already exists with other content: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise StorageConflict(
                    f"immutable manifest already exists with other content: {path}"
                ) from None
        _fsync_directory(path.parent)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _external_link(cas: CAS, source: Path, digest: str, logical_path: str | Path, mode: ImportMode) -> Path:
    target = cas._logical_target(logical_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve(strict=True).is_relative_to(cas.refs_path.resolve(strict=True)):
        from .cas import UnsafeStoragePath

        raise UnsafeStoragePath("logical path must remain below refs")

    with cas.locks.acquire(digest):
        if target.exists() or target.is_symlink():
            if mode is ImportMode.HARDLINK:
                try:
                    if os.path.samefile(source, target):
                        return target
                except FileNotFoundError:
                    pass
            elif mode is ImportMode.SYMLINK and target.is_symlink():
                if target.resolve(strict=True) == source:
                    return target
            raise StorageConflict(f"logical path already exists: {logical_path}")
        try:
            if mode is ImportMode.HARDLINK:
                os.link(source, target)
            else:
                target.symlink_to(os.path.relpath(source, target.parent))
        except FileExistsError as error:
            raise StorageConflict(f"logical path already exists: {logical_path}") from error
        _fsync_directory(target.parent)
    return target


def import_external(
    cas: CAS,
    source: Path,
    *,
    mode: ImportMode | str,
    logical_path: str | Path,
    expected_sha256: str | None = None,
) -> ExternalImport:
    selected_mode = ImportMode(mode)
    requested_source = Path(source).expanduser()
    try:
        source_path = requested_source.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"external source does not exist: {source}") from error
    if not source_path.is_file():
        raise ValueError("external source must be a regular file")

    # Validate the logical path before hashing or mutating the source.
    cas._logical_target(logical_path)
    digest, size = _hash_path(source_path)
    if expected_sha256 is not None:
        expected_sha256 = validate_digest(expected_sha256)
        if digest != expected_sha256:
            from .cas import CASIntegrityError

            raise CASIntegrityError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )

    blob: CASBlob | None = None
    if selected_mode in {ImportMode.COPY, ImportMode.MOVE}:
        blob = cas.put(source_path, expected_sha256=digest)
        storage_path = cas.link(blob.digest, logical_path)
    elif selected_mode in {ImportMode.SYMLINK, ImportMode.HARDLINK}:
        storage_path = _external_link(
            cas, source_path, digest, logical_path, selected_mode
        )
    else:
        storage_path = source_path

    identity_payload = {
        "logical_path": Path(logical_path).as_posix(),
        "mode": selected_mode.value,
        "source_path": str(source_path),
    }
    manifest_id = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    manifest_path = cas.manifests_path / "imports" / f"{manifest_id}.json"
    manifest = {
        "schema": "comfyng.external-import/v1",
        **identity_payload,
        "sha256": digest,
        "size_bytes": size,
        "storage_path": str(storage_path),
        "blob_uri": None if blob is None else blob.uri,
    }
    write_immutable_json(manifest_path, manifest)

    if selected_mode is ImportMode.MOVE:
        source_path.unlink()
        _fsync_directory(source_path.parent)

    return ExternalImport(
        mode=selected_mode,
        digest=digest,
        size_bytes=size,
        source_path=source_path,
        storage_path=storage_path,
        manifest_path=manifest_path,
        blob=blob,
    )


__all__ = [
    "ExternalImport",
    "ImportMode",
    "canonical_json_bytes",
    "import_external",
    "write_immutable_json",
]
