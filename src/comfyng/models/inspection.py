from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from comfyng.core.json_values import freeze_json_value


_MAX_SAFETENSORS_HEADER = 16 * 1024 * 1024
_MAX_JSON_DOCUMENT = 8 * 1024 * 1024
_HASH_CHUNK = 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "C64": 8,
    "C128": 16,
}


class ModelInspectionError(ValueError):
    """Raised when model files cannot be inspected safely and deterministically."""


@dataclass(frozen=True, slots=True)
class TensorInspection:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ModelFileInspection:
    path: Path
    sha256: str
    size_bytes: int
    format: str
    tensors: tuple[TensorInspection, ...]
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ModelInspection:
    files: tuple[ModelFileInspection, ...]
    aggregate_sha256: str
    total_size_bytes: int
    tensor_shapes: Mapping[str, tuple[int, ...]]
    tensor_dtypes: Mapping[str, str]
    safetensors_metadata: Mapping[str, str]
    config: Mapping[str, Any]
    repository_manifest: Mapping[str, Any]
    provider_declaration: Mapping[str, Any]


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_mapping(
    value: Mapping[str, Any] | Path | None,
    *,
    field: str,
) -> Mapping[str, Any]:
    if value is None:
        payload: Any = {}
    elif isinstance(value, Path):
        try:
            size = value.stat().st_size
        except OSError as error:
            raise ModelInspectionError(f"cannot read {field}: {error}") from error
        if size > _MAX_JSON_DOCUMENT:
            raise ModelInspectionError(f"{field} is too large")
        try:
            payload = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ModelInspectionError(f"invalid {field}: {error}") from error
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ModelInspectionError(f"{field} must be a JSON object or Path")
    if not isinstance(payload, dict):
        raise ModelInspectionError(f"{field} must be a JSON object")
    try:
        return freeze_json_value(payload, path=f"$.{field}")
    except ValueError as error:
        raise ModelInspectionError(f"invalid {field}: {error}") from error


def _parse_safetensors(path: Path, size: int) -> tuple[tuple[TensorInspection, ...], Mapping[str, str]]:
    if size < 10:
        raise ModelInspectionError(f"invalid safetensors file {path}: truncated header")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size <= 1 or header_size > _MAX_SAFETENSORS_HEADER:
            raise ModelInspectionError(
                f"invalid safetensors file {path}: header size is not bounded"
            )
        if header_size > size - 8:
            raise ModelInspectionError(
                f"invalid safetensors file {path}: truncated JSON header"
            )
        header_bytes = stream.read(header_size)
    try:
        header = json.loads(header_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ModelInspectionError(
            f"invalid safetensors file {path}: malformed JSON header"
        ) from error
    if not isinstance(header, dict):
        raise ModelInspectionError(
            f"invalid safetensors file {path}: header must be an object"
        )

    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise ModelInspectionError(
            f"invalid safetensors file {path}: metadata must contain strings"
        )
    metadata = freeze_json_value(raw_metadata, path="$.safetensors_metadata")
    data_size = size - 8 - header_size
    tensors: list[TensorInspection] = []
    for name, raw in sorted(header.items()):
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ModelInspectionError(
                f"invalid safetensors file {path}: malformed tensor entry"
            )
        dtype = raw.get("dtype")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ModelInspectionError(f"invalid dtype for tensor {name}")
        if dtype not in _DTYPE_BYTES:
            raise ModelInspectionError(f"unknown dtype for tensor {name}: {dtype}")
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ModelInspectionError(f"invalid shape for tensor {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise ModelInspectionError(f"invalid data offsets for tensor {name}")
        tensors.append(
            TensorInspection(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                data_offsets=(offsets[0], offsets[1]),
            )
        )
    if not tensors:
        raise ModelInspectionError(f"safetensors file {path} contains no tensors")
    cursor = 0
    for tensor in sorted(tensors, key=lambda item: item.data_offsets):
        start, end = tensor.data_offsets
        if start != cursor:
            raise ModelInspectionError(
                f"invalid safetensors file {path}: tensor data overlap or gap"
            )
        expected_size = math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
        if end - start != expected_size:
            raise ModelInspectionError(
                f"invalid byte length for tensor {tensor.name}"
            )
        cursor = end
    if cursor != data_size:
        raise ModelInspectionError(
            f"invalid safetensors file {path}: unclaimed tensor data"
        )
    return tuple(tensors), metadata


class ModelInspector:
    def inspect(
        self,
        files: Sequence[Path],
        *,
        config: Mapping[str, Any] | Path | None = None,
        repository_manifest: Mapping[str, Any] | Path | None = None,
        provider_declaration: Mapping[str, Any] | None = None,
    ) -> ModelInspection:
        if not files:
            raise ModelInspectionError("at least one model file is required")
        resolved: list[Path] = []
        for candidate in files:
            try:
                path = Path(candidate).expanduser().resolve(strict=True)
            except OSError as error:
                raise ModelInspectionError(f"model file does not exist: {candidate}") from error
            if not path.is_file():
                raise ModelInspectionError(f"model path is not a regular file: {candidate}")
            resolved.append(path)
        if len(set(resolved)) != len(resolved):
            raise ModelInspectionError("duplicate model file paths are not allowed")

        inspected: list[ModelFileInspection] = []
        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, str] = {}
        metadata_values: dict[str, set[str]] = {}
        for path in sorted(resolved):
            digest, size = _hash_file(path)
            tensors, metadata = _parse_safetensors(path, size)
            for tensor in tensors:
                previous_shape = shapes.get(tensor.name)
                if previous_shape is not None:
                    raise ModelInspectionError(
                        f"duplicate tensor across shards: {tensor.name}"
                    )
                shapes[tensor.name] = tensor.shape
                dtypes[tensor.name] = tensor.dtype
            for key, value in metadata.items():
                metadata_values.setdefault(key, set()).add(value)
            inspected.append(
                ModelFileInspection(
                    path=path,
                    sha256=digest,
                    size_bytes=size,
                    format="safetensors",
                    tensors=tensors,
                    metadata=metadata,
                )
            )

        if len(inspected) == 1:
            aggregate = inspected[0].sha256
        else:
            aggregate_hash = hashlib.sha256(b"comfyng-model/v1\0")
            for item in sorted(
                inspected, key=lambda candidate: (candidate.sha256, candidate.size_bytes)
            ):
                aggregate_hash.update(bytes.fromhex(item.sha256))
                aggregate_hash.update(item.size_bytes.to_bytes(16, "big"))
            aggregate = aggregate_hash.hexdigest()

        unambiguous_metadata = {
            key: next(iter(values))
            for key, values in metadata_values.items()
            if len(values) == 1
        }
        return ModelInspection(
            files=tuple(inspected),
            aggregate_sha256=aggregate,
            total_size_bytes=sum(item.size_bytes for item in inspected),
            tensor_shapes=freeze_json_value(shapes, path="$.tensor_shapes"),
            tensor_dtypes=freeze_json_value(dtypes, path="$.tensor_dtypes"),
            safetensors_metadata=freeze_json_value(
                unambiguous_metadata, path="$.safetensors_metadata"
            ),
            config=_json_mapping(config, field="config"),
            repository_manifest=_json_mapping(
                repository_manifest, field="repository_manifest"
            ),
            provider_declaration=_json_mapping(
                provider_declaration, field="provider_declaration"
            ),
        )


__all__ = [
    "ModelFileInspection",
    "ModelInspection",
    "ModelInspectionError",
    "ModelInspector",
    "TensorInspection",
]
