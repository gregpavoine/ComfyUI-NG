from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import BinaryIO, Any
from uuid import uuid4

from comfyng.core.json_values import freeze_json_value
from comfyng.database import Repositories

from .cas import CAS, _fsync_directory
from .imports import write_immutable_json


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    owner_type: str
    owner_id: str
    name: str
    version: int
    kind: str
    uri: str
    digest: str
    size_bytes: int
    manifest_path: Path
    metadata: Any
    job_id: str | None = None
    workflow_id: str | None = None
    created_at: str | None = None


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must contain valid Unicode") from error
    return value


class ArtifactStore:
    """Versioned artifact publication backed by SQLite and immutable CAS bytes."""

    def __init__(self, cas: CAS, repositories: Repositories) -> None:
        self.cas = cas
        self.repositories = repositories
        self._publication_lock = asyncio.Lock()

    def _manifest_path(self, artifact_id: str) -> Path:
        if not artifact_id or not artifact_id.isalnum():
            raise ValueError("artifact id must be alphanumeric")
        return self.cas.manifests_path / "artifacts" / f"{artifact_id}.json"

    @staticmethod
    def _from_row(row: dict[str, Any], manifest_path: Path) -> Artifact:
        metadata = row["metadata_json"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Artifact(
            id=row["id"],
            owner_type=row["owner_type"],
            owner_id=row["owner_id"],
            name=row["name"],
            version=row["version"],
            kind=row["kind"],
            uri=row["uri"],
            digest=row["sha256"],
            size_bytes=row["size_bytes"],
            manifest_path=manifest_path,
            metadata=freeze_json_value(metadata, path="$.metadata"),
            job_id=row["job_id"],
            workflow_id=row["workflow_id"],
            created_at=row["created_at"],
        )

    async def publish(
        self,
        *,
        owner_type: str,
        owner_id: str,
        name: str,
        kind: str,
        source: bytes | bytearray | memoryview | Path | BinaryIO,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        workflow_id: str | None = None,
    ) -> Artifact:
        owner_type = _required_text(owner_type, "owner_type")
        owner_id = _required_text(owner_id, "owner_id")
        name = _required_text(name, "name")
        kind = _required_text(kind, "kind")
        payload_metadata = {} if metadata is None else dict(metadata)
        freeze_json_value(payload_metadata, path="$.metadata")

        blob = self.cas.put(source)
        artifact_id = uuid4().hex
        reference_id = f"artifact-{artifact_id}"
        manifest_path = self._manifest_path(artifact_id)
        reference_added = False
        manifest_added = False
        try:
            async with self._publication_lock:
                async with self.repositories.transaction() as repositories:
                    row = await repositories.artifacts.create_version(
                        owner_type=owner_type,
                        owner_id=owner_id,
                        name=name,
                        kind=kind,
                        uri=blob.uri,
                        artifact_id=artifact_id,
                        job_id=job_id,
                        workflow_id=workflow_id,
                        sha256=blob.digest,
                        size_bytes=blob.size_bytes,
                        metadata_json=payload_metadata,
                    )
                    self.cas.add_reference(
                        reference_id,
                        {blob.digest},
                        metadata={"kind": "artifact", "artifact_id": artifact_id},
                    )
                    reference_added = True
                    write_immutable_json(
                        manifest_path,
                        {
                            "schema": "comfyng.artifact/v1",
                            "id": artifact_id,
                            "owner_type": owner_type,
                            "owner_id": owner_id,
                            "name": name,
                            "version": row["version"],
                            "kind": kind,
                            "uri": blob.uri,
                            "sha256": blob.digest,
                            "size_bytes": blob.size_bytes,
                            "job_id": job_id,
                            "workflow_id": workflow_id,
                            "metadata": payload_metadata,
                        },
                    )
                    manifest_added = True
            return self._from_row(row, manifest_path)
        except BaseException:
            if manifest_added:
                manifest_path.unlink(missing_ok=True)
                _fsync_directory(manifest_path.parent)
            if reference_added:
                self.cas.remove_reference(reference_id)
            raise

    async def get(self, artifact_id: str) -> Artifact | None:
        row = await self.repositories.artifacts.get(artifact_id)
        if row is None:
            return None
        manifest_path = self._manifest_path(artifact_id)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest is missing: {artifact_id}")
        return self._from_row(row, manifest_path)

    def open(self, artifact: Artifact) -> BinaryIO:
        return self.cas.open(artifact.digest)


__all__ = ["Artifact", "ArtifactStore"]
