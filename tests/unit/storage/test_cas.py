from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import os
from pathlib import Path

import pytest


class BrokenReader(BytesIO):
    def read(self, size: int = -1) -> bytes:
        if self.tell() >= 4:
            raise OSError("source disappeared")
        return super().read(min(size, 4))


def test_put_deduplicates_the_same_sha256_under_thread_concurrency(
    tmp_path: Path,
) -> None:
    from comfyng.storage.cas import CAS

    root = tmp_path / "storage"
    payload = (b"comfyui-ng-content-addressed-storage" * 4096) + b"!"
    expected_digest = hashlib.sha256(payload).hexdigest()

    with ThreadPoolExecutor(max_workers=12) as executor:
        blobs = list(executor.map(lambda _: CAS(root).put(payload), range(36)))

    assert {blob.digest for blob in blobs} == {expected_digest}
    assert {blob.size_bytes for blob in blobs} == {len(payload)}
    assert {blob.path for blob in blobs} == {
        root.resolve() / "blobs" / "sha256" / expected_digest
    }
    assert [path.name for path in (root / "blobs" / "sha256").iterdir()] == [
        expected_digest
    ]
    assert list((root / "partials").iterdir()) == []
    with CAS(root).open(expected_digest) as stream:
        assert stream.read() == payload


def test_put_uses_atomic_replace_and_cleans_partial_files_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from comfyng.storage.cas import CAS

    cas = CAS(tmp_path / "storage")
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        replaced.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", recording_replace)
    blob = cas.put(BytesIO(b"atomic payload"))

    assert replaced == [(replaced[0][0], blob.path)]
    assert replaced[0][0].parent == cas.partials_path
    assert not replaced[0][0].exists()

    with pytest.raises(OSError, match="source disappeared"):
        cas.put(BrokenReader(b"0123456789"))
    assert list(cas.partials_path.iterdir()) == []


def test_put_rejects_a_wrong_expected_digest_without_publishing(
    tmp_path: Path,
) -> None:
    from comfyng.storage.cas import CAS, CASIntegrityError

    cas = CAS(tmp_path / "storage")

    with pytest.raises(CASIntegrityError, match="SHA-256"):
        cas.put(b"payload", expected_sha256="0" * 64)

    assert list(cas.blobs_path.iterdir()) == []
    assert list(cas.partials_path.iterdir()) == []


@pytest.mark.parametrize(
    "logical_path",
    (
        "../escape",
        "models/../../escape",
        "/absolute/escape",
        "",
        ".",
    ),
)
def test_link_rejects_path_traversal_and_absolute_paths(
    tmp_path: Path,
    logical_path: str,
) -> None:
    from comfyng.storage.cas import CAS, UnsafeStoragePath

    cas = CAS(tmp_path / "storage")
    blob = cas.put(b"safe")

    with pytest.raises(UnsafeStoragePath):
        cas.link(blob.digest, logical_path)

    assert not (tmp_path / "escape").exists()


def test_link_is_idempotent_but_never_overwrites_other_content(tmp_path: Path) -> None:
    from comfyng.storage.cas import CAS, StorageConflict

    cas = CAS(tmp_path / "storage")
    first = cas.put(b"first")
    second = cas.put(b"second")

    target = cas.link(first.digest, "models/flux/model.safetensors")
    assert target.is_file()
    assert target.read_bytes() == b"first"
    assert cas.link(first.digest, "models/flux/model.safetensors") == target

    with pytest.raises(StorageConflict, match="already exists"):
        cas.link(second.digest, "models/flux/model.safetensors")

    assert target.read_bytes() == b"first"
