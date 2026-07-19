from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

import pytest


async def _store(tmp_path: Path):
    from comfyng.database import Database, Repositories
    from comfyng.storage.artifacts import ArtifactStore
    from comfyng.storage.cas import CAS

    database = await Database(tmp_path / "state.db").open()
    cas = CAS(tmp_path / "storage")
    return database, cas, ArtifactStore(cas, Repositories(database))


def test_artifacts_are_versioned_deduplicated_and_reference_tracked(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, cas, store = await _store(tmp_path)

        first = await store.publish(
            owner_type="workflow",
            owner_id="wf-demo",
            name="preview",
            kind="image/png",
            source=b"same-image",
            metadata={"width": 1024, "height": 1024},
        )
        second = await store.publish(
            owner_type="workflow",
            owner_id="wf-demo",
            name="preview",
            kind="image/png",
            source=b"same-image",
        )

        assert (first.version, second.version) == (1, 2)
        assert first.digest == second.digest
        assert first.uri == f"cas://sha256/{first.digest}"
        assert await store.get(first.id) == first
        with store.open(first) as stream:
            assert stream.read() == b"same-image"
        assert cas.references_for(first.digest) == frozenset(
            {f"artifact-{first.id}", f"artifact-{second.id}"}
        )
        assert len(list(cas.blobs_path.iterdir())) == 1
        manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "comfyng.artifact/v1"
        assert manifest["version"] == 1
        assert manifest["metadata"] == {"height": 1024, "width": 1024}
        await database.close()

    asyncio.run(scenario())


def test_concurrent_artifact_publication_assigns_unique_monotonic_versions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, _, store = await _store(tmp_path)

        artifacts = await asyncio.gather(
            *(
                store.publish(
                    owner_type="job",
                    owner_id="job-demo",
                    name="frame",
                    kind="image/png",
                    source=f"frame-{index}".encode(),
                )
                for index in range(20)
            )
        )

        assert sorted(artifact.version for artifact in artifacts) == list(range(1, 21))
        assert len({artifact.id for artifact in artifacts}) == 20
        await database.close()

    asyncio.run(scenario())


def test_failed_database_publication_leaves_only_collectable_orphan_blob(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database, cas, store = await _store(tmp_path)

        with pytest.raises(sqlite3.IntegrityError):
            await store.publish(
                owner_type="job",
                owner_id="missing-job",
                name="preview",
                kind="image/png",
                source=b"orphan-after-rollback",
                job_id="missing-job",
            )

        assert cas.iter_references() == ()
        assert list((cas.manifests_path / "artifacts").glob("*.json")) == []
        async with database.connection() as connection:
            count = await connection.execute_fetchall("SELECT COUNT(*) FROM artifacts")
        assert count[0][0] == 0
        await database.close()

    asyncio.run(scenario())
