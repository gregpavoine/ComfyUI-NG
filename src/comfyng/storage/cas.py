from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO, Literal
import json
import re

from .locks import DigestLockPool, validate_digest


_CHUNK_SIZE = 1024 * 1024


class StorageError(RuntimeError):
    """Base class for local content-addressed storage failures."""


class CASIntegrityError(StorageError):
    """Raised when bytes do not match the requested content identity."""


class UnsafeStoragePath(StorageError, ValueError):
    """Raised when a logical path could leave its managed storage root."""


class StorageConflict(StorageError):
    """Raised when an immutable logical path already contains other bytes."""


class ReferenceManifestError(StorageError):
    """Raised when reference metadata is invalid; collection must fail closed."""


@dataclass(frozen=True, slots=True)
class CASBlob:
    digest: str
    size_bytes: int
    path: Path

    @property
    def uri(self) -> str:
        return f"cas://sha256/{self.digest}"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class CAS:
    """Immutable SHA-256 blobs plus traversal-safe logical references."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve(strict=False)
        self.blobs_path = self.root / "blobs" / "sha256"
        self.manifests_path = self.root / "manifests"
        self.refs_path = self.root / "refs"
        self.metadata_path = self.root / "metadata"
        self.thumbnails_path = self.root / "thumbnails"
        self.partials_path = self.root / "partials"
        self.locks_path = self.root / ".locks"
        for path in (
            self.blobs_path,
            self.manifests_path,
            self.refs_path,
            self.metadata_path,
            self.thumbnails_path,
            self.partials_path,
            self.locks_path,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.locks = DigestLockPool(self.locks_path)

    def blob_path(self, digest: str) -> Path:
        return self.blobs_path / validate_digest(digest)

    def put(
        self,
        source: bytes | bytearray | memoryview | Path | BinaryIO,
        *,
        expected_sha256: str | None = None,
    ) -> CASBlob:
        if expected_sha256 is not None:
            expected_sha256 = validate_digest(expected_sha256)

        owned_stream: BinaryIO | None = None
        if isinstance(source, (bytes, bytearray, memoryview)):
            from io import BytesIO

            stream: BinaryIO = BytesIO(bytes(source))
        elif isinstance(source, Path):
            owned_stream = source.open("rb")
            stream = owned_stream
        elif hasattr(source, "read"):
            stream = source
        else:
            raise TypeError("source must be bytes, a Path, or a binary stream")

        descriptor, partial_name = tempfile.mkstemp(
            prefix="put-", suffix=".partial", dir=self.partials_path
        )
        partial = Path(partial_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as target:
                while chunk := stream.read(_CHUNK_SIZE):
                    if not isinstance(chunk, bytes):
                        raise TypeError("binary sources must return bytes")
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                target.flush()
                os.fsync(target.fileno())

            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise CASIntegrityError(
                    f"SHA-256 mismatch: expected {expected_sha256}, got {actual}"
                )

            destination = self.blob_path(actual)
            with self.locks.acquire(actual):
                if destination.exists():
                    existing_digest, existing_size = _hash_path(destination)
                    if existing_digest != actual:
                        raise CASIntegrityError(
                            f"published blob {actual} no longer matches its SHA-256"
                        )
                    return CASBlob(actual, existing_size, destination)
                os.chmod(partial, 0o444)
                os.replace(partial, destination)
                _fsync_directory(self.blobs_path)
            return CASBlob(actual, size, destination)
        finally:
            if owned_stream is not None:
                owned_stream.close()
            partial.unlink(missing_ok=True)

    def open(self, digest: str) -> BinaryIO:
        path = self.blob_path(digest)
        try:
            return path.open("rb")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"CAS blob {digest} does not exist") from error

    def _logical_target(self, logical_path: str | Path) -> Path:
        raw = os.fspath(logical_path)
        if not raw or raw == "." or "\x00" in raw:
            raise UnsafeStoragePath("logical path must identify a file below refs")
        relative = Path(raw)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise UnsafeStoragePath("logical path must remain below refs")
        target = self.refs_path.joinpath(relative)
        resolved = target.resolve(strict=False)
        if not resolved.is_relative_to(self.refs_path.resolve(strict=True)):
            raise UnsafeStoragePath("logical path must remain below refs")
        return target

    def link(
        self,
        digest: str,
        logical_path: str | Path,
        *,
        mode: Literal["hardlink", "copy", "symlink"] = "hardlink",
    ) -> Path:
        source = self.blob_path(digest)
        if not source.is_file():
            raise FileNotFoundError(f"CAS blob {digest} does not exist")
        target = self._logical_target(logical_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve(strict=True).is_relative_to(
            self.refs_path.resolve(strict=True)
        ):
            raise UnsafeStoragePath("logical path must remain below refs")

        with self.locks.acquire(digest):
            if target.exists() or target.is_symlink():
                if target.is_file() and _hash_path(target)[0] == digest:
                    return target
                raise StorageConflict(f"logical path already exists: {logical_path}")
            try:
                if mode == "hardlink":
                    os.link(source, target)
                elif mode == "symlink":
                    target.symlink_to(os.path.relpath(source, target.parent))
                elif mode == "copy":
                    with source.open("rb") as stream, target.open("xb") as output:
                        shutil.copyfileobj(stream, output, _CHUNK_SIZE)
                        output.flush()
                        os.fsync(output.fileno())
                else:
                    raise ValueError(f"unsupported link mode: {mode}")
            except FileExistsError as error:
                if target.is_file() and _hash_path(target)[0] == digest:
                    return target
                raise StorageConflict(
                    f"logical path already exists: {logical_path}"
                ) from error
            _fsync_directory(target.parent)
        return target

    def import_external(
        self,
        source: Path,
        *,
        mode: str,
        logical_path: str | Path,
        expected_sha256: str | None = None,
    ):
        from .imports import import_external

        return import_external(
            self,
            source,
            mode=mode,
            logical_path=logical_path,
            expected_sha256=expected_sha256,
        )

    def _reference_path(self, reference_id: str) -> Path:
        if (
            not isinstance(reference_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", reference_id) is None
        ):
            raise UnsafeStoragePath("reference id is not a safe storage name")
        return self.metadata_path / "references" / f"{reference_id}.json"

    def _read_reference(self, path: Path) -> tuple[str, frozenset[str]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest root must be an object")
            if payload.get("schema") != "comfyng.cas-reference/v1":
                raise ValueError("unknown reference schema")
            reference_id = payload["reference_id"]
            if path != self._reference_path(reference_id):
                raise ValueError("reference id does not match filename")
            values = payload["digests"]
            if not isinstance(values, list) or not values:
                raise ValueError("digests must be a non-empty list")
            digests = frozenset(validate_digest(value) for value in values)
            if len(digests) != len(values) or values != sorted(values):
                raise ValueError("digests must be unique and sorted")
            if not isinstance(payload.get("metadata", {}), dict):
                raise ValueError("metadata must be an object")
            return reference_id, digests
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ReferenceManifestError(
                f"invalid CAS reference manifest {path}: {error}"
            ) from error

    def add_reference(
        self,
        reference_id: str,
        digests: Iterable[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        from .imports import write_immutable_json

        path = self._reference_path(reference_id)
        selected = frozenset(validate_digest(digest) for digest in digests)
        if not selected:
            raise ValueError("a reference must contain at least one digest")
        payload = {
            "schema": "comfyng.cas-reference/v1",
            "reference_id": reference_id,
            "digests": sorted(selected),
            "metadata": {} if metadata is None else dict(metadata),
        }
        with self.locks.acquire_many(selected):
            missing = [digest for digest in selected if not self.blob_path(digest).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"cannot reference missing CAS blobs: {', '.join(sorted(missing))}"
                )
            return write_immutable_json(path, payload)

    def remove_reference(self, reference_id: str) -> bool:
        path = self._reference_path(reference_id)
        if not path.exists():
            return False
        _, digests = self._read_reference(path)
        with self.locks.acquire_many(digests):
            if not path.exists():
                return False
            path.unlink()
            _fsync_directory(path.parent)
        return True

    def iter_references(self) -> tuple[tuple[str, frozenset[str]], ...]:
        directory = self.metadata_path / "references"
        if not directory.exists():
            return ()
        return tuple(
            self._read_reference(path)
            for path in sorted(directory.glob("*.json"))
        )

    def references_for(self, digest: str) -> frozenset[str]:
        digest = validate_digest(digest)
        return frozenset(
            reference_id
            for reference_id, digests in self.iter_references()
            if digest in digests
        )

    def all_referenced_digests(self) -> frozenset[str]:
        return frozenset(
            digest
            for _, digests in self.iter_references()
            for digest in digests
        )


__all__ = [
    "CAS",
    "CASBlob",
    "CASIntegrityError",
    "ReferenceManifestError",
    "StorageConflict",
    "StorageError",
    "UnsafeStoragePath",
]
