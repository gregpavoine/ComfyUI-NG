from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import pytest


def test_references_are_immutable_idempotent_and_queryable(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS, StorageConflict, UnsafeStoragePath

    cas = CAS(tmp_path / "storage")
    first = cas.put(b"first")
    second = cas.put(b"second")

    manifest = cas.add_reference(
        "model-demo",
        {first.digest, second.digest},
        metadata={"kind": "model", "version": 1},
    )
    assert cas.add_reference(
        "model-demo",
        {second.digest, first.digest},
        metadata={"version": 1, "kind": "model"},
    ) == manifest
    assert cas.references_for(first.digest) == frozenset({"model-demo"})
    assert cas.references_for(second.digest) == frozenset({"model-demo"})

    with pytest.raises(StorageConflict, match="immutable manifest"):
        cas.add_reference("model-demo", {first.digest})
    with pytest.raises(UnsafeStoragePath):
        cas.add_reference("../outside", {first.digest})

    assert cas.remove_reference("model-demo") is True
    assert cas.remove_reference("model-demo") is False
    assert cas.references_for(first.digest) == frozenset()


def test_gc_deletes_only_unreferenced_blobs_and_supports_dry_run(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS
    from comfyng.storage.gc import GarbageCollector

    cas = CAS(tmp_path / "storage")
    kept = cas.put(b"kept")
    removed = cas.put(b"removed")
    cas.add_reference("artifact-preview", {kept.digest})
    collector = GarbageCollector(cas)

    dry_run = collector.collect(dry_run=True, min_age_seconds=0)
    assert dry_run.deleted_digests == (removed.digest,)
    assert dry_run.reclaimed_bytes == len(b"removed")
    assert removed.path.exists()

    report = collector.collect(min_age_seconds=0)
    assert report.deleted_digests == (removed.digest,)
    assert report.reclaimed_bytes == len(b"removed")
    assert kept.path.read_bytes() == b"kept"
    assert not removed.path.exists()


def test_gc_cleans_only_stale_partials(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS
    from comfyng.storage.gc import GarbageCollector

    cas = CAS(tmp_path / "storage")
    stale = cas.partials_path / "stale.partial"
    fresh = cas.partials_path / "fresh.partial"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    old = time.time() - 7_200
    os.utime(stale, (old, old))

    report = GarbageCollector(cas).cleanup_partials(max_age_seconds=3_600)

    assert report.deleted_paths == (stale,)
    assert report.reclaimed_bytes == 3
    assert fresh.read_bytes() == b"new"


def test_reference_publication_wins_race_with_gc_under_digest_lock(
    tmp_path: Path,
) -> None:
    from comfyng.storage.cas import CAS
    from comfyng.storage.gc import GarbageCollector

    cas = CAS(tmp_path / "storage")
    blob = cas.put(b"raced")
    collector = GarbageCollector(cas)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with cas.locks.acquire(blob.digest):
            future = executor.submit(
                collector.collect, min_age_seconds=0
            )
            cas.add_reference("model-raced", {blob.digest})
        report = future.result(timeout=10)

    assert report.deleted_digests == ()
    assert blob.path.read_bytes() == b"raced"


def test_gc_fails_closed_on_a_corrupt_reference_manifest(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS
    from comfyng.storage.gc import GarbageCollector, ReferenceManifestError

    cas = CAS(tmp_path / "storage")
    blob = cas.put(b"must-survive")
    reference_dir = cas.metadata_path / "references"
    reference_dir.mkdir(parents=True)
    (reference_dir / "corrupt.json").write_text(
        json.dumps({"schema": "wrong", "digests": [blob.digest]}),
        encoding="utf-8",
    )

    with pytest.raises(ReferenceManifestError):
        GarbageCollector(cas).collect(min_age_seconds=0)

    assert blob.path.read_bytes() == b"must-survive"
