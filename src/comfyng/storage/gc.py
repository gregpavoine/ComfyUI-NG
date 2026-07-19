from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

from .cas import CAS, ReferenceManifestError, _fsync_directory
from .locks import InvalidDigest, validate_digest


@dataclass(frozen=True, slots=True)
class GCReport:
    deleted_digests: tuple[str, ...]
    reclaimed_bytes: int
    dry_run: bool


@dataclass(frozen=True, slots=True)
class PartialCleanupReport:
    deleted_paths: tuple[Path, ...]
    reclaimed_bytes: int


class GarbageCollector:
    def __init__(self, cas: CAS) -> None:
        self.cas = cas

    def collect(
        self,
        *,
        dry_run: bool = False,
        min_age_seconds: float = 24 * 60 * 60,
    ) -> GCReport:
        if min_age_seconds < 0:
            raise ValueError("min_age_seconds must be non-negative")

        # Validate every reference before considering the first deletion.
        self.cas.all_referenced_digests()
        cutoff = time.time() - min_age_seconds
        deleted: list[str] = []
        reclaimed = 0
        for path in sorted(self.cas.blobs_path.iterdir()):
            if not path.is_file():
                continue
            try:
                digest = validate_digest(path.name)
            except InvalidDigest:
                continue
            initial = path.stat()
            if initial.st_mtime > cutoff:
                continue
            with self.cas.locks.acquire(digest):
                if not path.exists() or self.cas.references_for(digest):
                    continue
                size = path.stat().st_size
                deleted.append(digest)
                reclaimed += size
                if not dry_run:
                    path.unlink()
                    _fsync_directory(self.cas.blobs_path)
        return GCReport(tuple(deleted), reclaimed, dry_run)

    def cleanup_partials(
        self,
        *,
        max_age_seconds: float = 24 * 60 * 60,
    ) -> PartialCleanupReport:
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        cutoff = time.time() - max_age_seconds
        deleted: list[Path] = []
        reclaimed = 0
        for path in sorted(self.cas.partials_path.glob("*.partial")):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if not path.is_file() or stat.st_mtime > cutoff:
                continue
            reclaimed += stat.st_size
            path.unlink(missing_ok=True)
            deleted.append(path)
        if deleted:
            _fsync_directory(self.cas.partials_path)
        return PartialCleanupReport(tuple(deleted), reclaimed)


__all__ = [
    "GCReport",
    "GarbageCollector",
    "PartialCleanupReport",
    "ReferenceManifestError",
]
