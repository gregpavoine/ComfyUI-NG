from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path


def _put_in_process(root: str, payload: bytes) -> tuple[str, int, str]:
    from comfyng.storage.cas import CAS

    blob = CAS(Path(root)).put(payload)
    return blob.digest, blob.size_bytes, str(blob.path)


def test_processes_publish_one_blob_for_the_same_digest(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS

    root = tmp_path / "shared-storage"
    payload = b"multiprocess-deduplication" * 4096
    expected = hashlib.sha256(payload).hexdigest()
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        results = [
            future.result(timeout=30)
            for future in (
                executor.submit(_put_in_process, str(root), payload)
                for _ in range(12)
            )
        ]

    cas = CAS(root)
    assert results == [
        (expected, len(payload), str(cas.blob_path(expected)))
    ] * 12
    assert [path.name for path in cas.blobs_path.iterdir()] == [expected]
    assert list(cas.partials_path.iterdir()) == []
