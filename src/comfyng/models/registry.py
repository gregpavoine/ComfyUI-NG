from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import UUID, uuid4

from comfyng.core.json_values import freeze_json_value
from comfyng.database import Repositories
from comfyng.storage.cas import CAS, UnsafeStoragePath, _fsync_directory
from comfyng.storage.imports import (
    ExternalImport,
    ImportMode,
    canonical_json_bytes,
)

from .capabilities import ModelCapabilities, ModelHandle
from .detector import (
    ArchitectureDetection,
    ArchitectureDetector,
    ArchitectureEvidence,
)
from .inspection import ModelFileInspection, ModelInspector
from .legacy import require_modern


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: Path
    logical_name: str | None = None
    kind: str = "weights"

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("model file kind must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ModelSource:
    provider: str
    source_id: str
    revision: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("provider", "source_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.revision, str):
            raise ValueError("revision must be a string")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_value(self.metadata, path="$.source.metadata"),
        )


@dataclass(frozen=True, slots=True)
class RegisteredModelFile:
    id: str
    kind: str
    logical_name: str
    path: Path
    source_path: Path
    digest: str
    size_bytes: int
    format: str
    import_mode: ImportMode


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    handle: ModelHandle
    status: str
    files: tuple[RegisteredModelFile, ...]
    manifest_path: Path
    detection: ArchitectureDetection


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must contain valid Unicode") from error
    return value


def _safe_logical_name(value: str) -> str:
    _required_text(value, "logical_name")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeStoragePath("model logical name must be a safe relative path")
    return path.as_posix()


def _publish_json_replace(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish a new immutable model manifest with fsync + atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"model manifest already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _remove_empty_parents(path: Path, stop: Path) -> None:
    parent = path.parent
    while parent != stop and parent.is_relative_to(stop):
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def _restore_moved_source(result: ExternalImport) -> None:
    if result.mode is not ImportMode.MOVE or result.source_path.exists():
        return
    result.source_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(result.storage_path, result.source_path)
    except OSError:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{result.source_path.name}.",
            suffix=".restore",
            dir=result.source_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as target, result.storage_path.open(
                "rb"
            ) as source:
                shutil.copyfileobj(source, target, 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, result.source_path)
        finally:
            temporary.unlink(missing_ok=True)
    _fsync_directory(result.source_path.parent)


class ModelRegistry:
    """Atomic SQLite/CAS publication for inspected modern model files."""

    def __init__(
        self,
        cas: CAS,
        repositories: Repositories,
        *,
        inspector: ModelInspector | None = None,
        detector: ArchitectureDetector | None = None,
    ) -> None:
        self.cas = cas
        self.repositories = repositories
        self.inspector = ModelInspector() if inspector is None else inspector
        self.detector = ArchitectureDetector() if detector is None else detector

    @staticmethod
    def _normalize_files(files: Sequence[Path | ModelFile]) -> tuple[ModelFile, ...]:
        if not files:
            raise ValueError("at least one model file is required")
        normalized: list[ModelFile] = []
        names: set[str] = set()
        for value in files:
            model_file = value if isinstance(value, ModelFile) else ModelFile(Path(value))
            logical_name = _safe_logical_name(
                model_file.logical_name or model_file.path.name
            )
            if logical_name in names:
                raise ValueError(f"duplicate model logical name: {logical_name}")
            names.add(logical_name)
            normalized.append(
                ModelFile(
                    path=model_file.path,
                    logical_name=logical_name,
                    kind=model_file.kind,
                )
            )
        return tuple(normalized)

    async def import_model(
        self,
        name: str,
        files: Sequence[Path | ModelFile],
        *,
        mode: ImportMode | str = ImportMode.COPY,
        config: Mapping[str, Any] | Path | None = None,
        repository_manifest: Mapping[str, Any] | Path | None = None,
        provider_declaration: Mapping[str, Any] | None = None,
        model_source: ModelSource | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RegisteredModel:
        name = _required_text(name, "name")
        selected_mode = ImportMode(mode)
        normalized = self._normalize_files(files)
        payload_metadata = {} if metadata is None else dict(metadata)
        freeze_json_value(payload_metadata, path="$.metadata")

        # Inspection and legacy refusal happen before storage, DB or runtime work.
        inspection = self.inspector.inspect(
            [item.path for item in normalized],
            config=config,
            repository_manifest=repository_manifest,
            provider_declaration=provider_declaration,
        )
        detection = require_modern(self.detector.detect(inspection))
        if detection.capabilities is None:
            raise RuntimeError("supported architecture has no capability contract")

        inspected_by_path: dict[Path, ModelFileInspection] = {
            item.path: item for item in inspection.files
        }
        model_id = uuid4()
        model_key = model_id.hex
        reference_id = f"model-{model_key}"
        logical_base = Path("models") / detection.family / model_key
        manifest_path = self.cas.manifests_path / "models" / f"{model_key}.json"
        imported: list[tuple[ModelFile, ModelFileInspection, ExternalImport]] = []
        reference_added = False
        model_manifest_added = False
        try:
            for model_file in normalized:
                resolved = model_file.path.expanduser().resolve(strict=True)
                inspected = inspected_by_path[resolved]
                external = self.cas.import_external(
                    resolved,
                    mode=selected_mode.value,
                    logical_path=logical_base / model_file.logical_name,
                    expected_sha256=inspected.sha256,
                )
                imported.append((model_file, inspected, external))

            source_payload = None
            if model_source is not None:
                source_payload = {
                    "provider": model_source.provider,
                    "source_id": model_source.source_id,
                    "revision": model_source.revision,
                    "metadata": model_source.metadata,
                }
            detection_payload = {
                "family": detection.family,
                "architecture": detection.architecture,
                "supported": detection.supported,
                "generation": detection.generation,
                "confidence": detection.confidence,
                "selected_source": detection.selected_source,
                "quantization": detection.quantization,
                "evidence": [
                    {
                        "source": item.source,
                        "family": item.family,
                        "architecture": item.architecture,
                        "score": item.score,
                        "detail": item.detail,
                    }
                    for item in detection.evidence
                ],
            }
            repository_payload = inspection.repository_manifest
            license_value = repository_payload.get("license", "unknown")
            if not isinstance(license_value, str) or not license_value.strip():
                license_value = "unknown"
            model_metadata = {
                **payload_metadata,
                "family": detection.family,
                "aggregate_sha256": inspection.aggregate_sha256,
                "quantization": detection.quantization,
                "license": license_value,
                "model_manifest": str(manifest_path),
                "detection": detection_payload,
            }
            capabilities_payload = detection.capabilities.to_builtins()
            manifest_payload = {
                "schema": "comfyng.model/v1",
                "id": model_key,
                "name": name,
                "status": "available",
                "family": detection.family,
                "architecture": detection.architecture,
                "aggregate_sha256": inspection.aggregate_sha256,
                "total_size_bytes": inspection.total_size_bytes,
                "license": license_value,
                "capabilities": capabilities_payload,
                "detection": detection_payload,
                "metadata": payload_metadata,
                "source": source_payload,
                "files": [
                    {
                        "kind": model_file.kind,
                        "logical_name": model_file.logical_name,
                        "source_path": str(external.source_path),
                        "path": str(external.storage_path),
                        "sha256": inspected.sha256,
                        "size_bytes": inspected.size_bytes,
                        "format": inspected.format,
                        "import_mode": external.mode.value,
                    }
                    for model_file, inspected, external in imported
                ],
            }

            registered_files: list[RegisteredModelFile] = []
            async with self.repositories.transaction() as repositories:
                await repositories.models.create(
                    {
                        "id": model_key,
                        "name": name,
                        "architecture": detection.architecture,
                        "status": "discovering",
                        "capabilities_json": capabilities_payload,
                        "metadata_json": model_metadata,
                    }
                )
                for model_file, inspected, external in imported:
                    file_id = uuid4().hex
                    await repositories.model_files.create(
                        {
                            "id": file_id,
                            "model_id": model_key,
                            "kind": model_file.kind,
                            "path": str(external.storage_path),
                            "sha256": inspected.sha256,
                            "size_bytes": inspected.size_bytes,
                            "format": inspected.format,
                            "metadata_json": {
                                "logical_name": model_file.logical_name,
                                "source_path": str(external.source_path),
                                "import_mode": external.mode.value,
                            },
                        }
                    )
                    registered_files.append(
                        RegisteredModelFile(
                            id=file_id,
                            kind=model_file.kind,
                            logical_name=model_file.logical_name,
                            path=external.storage_path,
                            source_path=external.source_path,
                            digest=inspected.sha256,
                            size_bytes=inspected.size_bytes,
                            format=inspected.format,
                            import_mode=external.mode,
                        )
                    )
                if model_source is not None:
                    await repositories.model_sources.create(
                        {
                            "id": uuid4().hex,
                            "model_id": model_key,
                            "provider": model_source.provider,
                            "source_id": model_source.source_id,
                            "revision": model_source.revision,
                            "metadata_json": model_source.metadata,
                        }
                    )
                managed_digests = {
                    external.blob.digest
                    for _, _, external in imported
                    if external.blob is not None
                }
                if managed_digests:
                    self.cas.add_reference(
                        reference_id,
                        managed_digests,
                        metadata={"kind": "model", "model_id": model_key},
                    )
                    reference_added = True
                _publish_json_replace(manifest_path, manifest_payload)
                model_manifest_added = True
                changed = await repositories.models.transition_state(
                    model_key,
                    expected=("discovering",),
                    target="available",
                )
                if not changed:
                    raise RuntimeError("model publication state transition was lost")

            primary = registered_files[0]
            handle = ModelHandle(
                id=model_id,
                family=detection.family,
                architecture=detection.architecture,
                local_path=primary.path.resolve(strict=False),
                sha256=inspection.aggregate_sha256,
                size_bytes=inspection.total_size_bytes,
                source_provider=None if model_source is None else model_source.provider,
                source_model_id=None if model_source is None else model_source.source_id,
                source_revision=None if model_source is None else model_source.revision or None,
                metadata=model_metadata,
            )
            return RegisteredModel(
                handle=handle,
                status="available",
                files=tuple(registered_files),
                manifest_path=manifest_path,
                detection=detection,
            )
        except BaseException:
            if model_manifest_added:
                manifest_path.unlink(missing_ok=True)
                _fsync_directory(manifest_path.parent)
            if reference_added:
                self.cas.remove_reference(reference_id)
            for _, _, external in reversed(imported):
                _restore_moved_source(external)
                if external.storage_path != external.source_path:
                    external.storage_path.unlink(missing_ok=True)
                    _remove_empty_parents(external.storage_path, self.cas.refs_path)
                external.manifest_path.unlink(missing_ok=True)
                if external.manifest_path.parent.exists():
                    _fsync_directory(external.manifest_path.parent)
            raise

    async def evict(self, model_id: UUID | str) -> bool:
        key = model_id.hex if isinstance(model_id, UUID) else UUID(str(model_id)).hex
        return await self.repositories.models.transition_state(
            key,
            expected=("available", "loading", "loaded", "offloaded"),
            target="evicted",
        )

    async def get(self, model_id: UUID | str) -> RegisteredModel | None:
        key = model_id.hex if isinstance(model_id, UUID) else UUID(str(model_id)).hex
        row = await self.repositories.models.get(key)
        if row is None:
            return None
        manifest_path = self.cas.manifests_path / "models" / f"{key}.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        capability_payload = json.loads(row["capabilities_json"])
        capabilities = ModelCapabilities.from_builtins(capability_payload)
        detection_payload = payload["detection"]
        detection = ArchitectureDetection(
            family=detection_payload["family"],
            architecture=detection_payload["architecture"],
            supported=detection_payload["supported"],
            generation=detection_payload["generation"],
            confidence=detection_payload["confidence"],
            selected_source=detection_payload["selected_source"],
            quantization=detection_payload["quantization"],
            evidence=tuple(
                ArchitectureEvidence(**item) for item in detection_payload["evidence"]
            ),
            capabilities=capabilities,
        )
        source = payload.get("source")
        file_rows = await self.repositories.model_files.list(
            filters={"model_id": key}, order_by="path"
        )
        rows_by_path = {row["path"]: row for row in file_rows}
        if len(rows_by_path) != len(payload["files"]):
            raise RuntimeError("model manifest and database file set differ")
        registered_files = tuple(
            RegisteredModelFile(
                id=rows_by_path[file_payload["path"]]["id"],
                kind=rows_by_path[file_payload["path"]]["kind"],
                logical_name=file_payload["logical_name"],
                path=Path(file_payload["path"]),
                source_path=Path(file_payload["source_path"]),
                digest=file_payload["sha256"],
                size_bytes=file_payload["size_bytes"],
                format=file_payload["format"],
                import_mode=ImportMode(file_payload["import_mode"]),
            )
            for file_payload in payload["files"]
        )
        metadata = json.loads(row["metadata_json"])
        handle = ModelHandle(
            id=UUID(key),
            family=detection.family,
            architecture=detection.architecture,
            local_path=registered_files[0].path.resolve(strict=False),
            sha256=payload["aggregate_sha256"],
            size_bytes=payload["total_size_bytes"],
            source_provider=None if source is None else source["provider"],
            source_model_id=None if source is None else source["source_id"],
            source_revision=None if source is None else source["revision"] or None,
            metadata=metadata,
        )
        return RegisteredModel(
            handle=handle,
            status=row["status"],
            files=registered_files,
            manifest_path=manifest_path,
            detection=detection,
        )

    async def list(self, *, status: str | None = None) -> tuple[RegisteredModel, ...]:
        filters = None if status is None else {"status": status}
        rows = await self.repositories.models.list(filters=filters, order_by="name")
        models = [await self.get(row["id"]) for row in rows]
        return tuple(model for model in models if model is not None)


__all__ = [
    "ModelFile",
    "ModelRegistry",
    "ModelSource",
    "RegisteredModel",
    "RegisteredModelFile",
]
