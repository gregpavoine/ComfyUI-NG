from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _safetensors_bytes(
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    *,
    metadata: dict[str, str] | None = None,
) -> bytes:
    header: dict[str, object] = {}
    offset = 0
    widths = {"F16": 2, "BF16": 2, "F32": 4, "I8": 1}
    for name, (dtype, shape) in tensors.items():
        length = widths[dtype]
        for dimension in shape:
            length *= dimension
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + length],
        }
        offset += length
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + bytes(offset)


def test_inspector_reads_bounded_safetensors_metadata_and_shapes_without_extension(
    tmp_path: Path,
) -> None:
    from comfyng.models.inspection import ModelInspector

    payload = _safetensors_bytes(
        {
            "transformer.double_blocks.0.img_attn.qkv.weight": (
                "BF16",
                (12, 8),
            ),
            "transformer.img_in.weight": ("F16", (8, 4)),
        },
        metadata={"comfyng.family": "flux1", "quantization": "none"},
    )
    path = tmp_path / "totally-renamed.data"
    path.write_bytes(payload)

    inspection = ModelInspector().inspect(
        [path],
        config={"model_type": "flux"},
        repository_manifest={"license": "apache-2.0"},
        provider_declaration={"family": "flux1"},
    )

    assert inspection.total_size_bytes == len(payload)
    assert inspection.aggregate_sha256 == hashlib.sha256(payload).hexdigest()
    assert inspection.files[0].format == "safetensors"
    assert inspection.files[0].sha256 == inspection.aggregate_sha256
    assert inspection.safetensors_metadata == {
        "comfyng.family": "flux1",
        "quantization": "none",
    }
    assert inspection.tensor_shapes[
        "transformer.double_blocks.0.img_attn.qkv.weight"
    ] == (12, 8)
    assert inspection.tensor_dtypes["transformer.img_in.weight"] == "F16"
    assert inspection.config == {"model_type": "flux"}
    assert inspection.repository_manifest == {"license": "apache-2.0"}
    assert inspection.provider_declaration == {"family": "flux1"}


def test_multifile_aggregate_hash_is_order_independent_and_paths_are_unique(
    tmp_path: Path,
) -> None:
    from comfyng.models.inspection import ModelInspectionError, ModelInspector

    first = tmp_path / "shard-a"
    second = tmp_path / "shard-b"
    first.write_bytes(_safetensors_bytes({"a": ("F16", (1,))}))
    second.write_bytes(_safetensors_bytes({"b": ("F16", (1,))}))
    inspector = ModelInspector()

    forward = inspector.inspect([first, second])
    reverse = inspector.inspect([second, first])

    assert forward.aggregate_sha256 == reverse.aggregate_sha256
    assert [item.path for item in forward.files] == sorted((first, second))
    with pytest.raises(ModelInspectionError, match="duplicate"):
        inspector.inspect([first, first])


@pytest.mark.parametrize(
    "payload",
    (
        b"not-a-safetensors-file",
        (64 * 1024 * 1024).to_bytes(8, "little") + b"{}",
        (2).to_bytes(8, "little") + b"[]",
    ),
)
def test_inspector_rejects_corrupt_or_excessive_safetensors_headers(
    tmp_path: Path,
    payload: bytes,
) -> None:
    from comfyng.models.inspection import ModelInspectionError, ModelInspector

    path = tmp_path / "weights"
    path.write_bytes(payload)

    with pytest.raises(ModelInspectionError):
        ModelInspector().inspect([path])


def test_inspector_loads_bounded_json_configuration_files(tmp_path: Path) -> None:
    from comfyng.models.inspection import ModelInspectionError, ModelInspector

    weights = tmp_path / "weights"
    weights.write_bytes(_safetensors_bytes({"tensor": ("F16", (1,))}))
    config = tmp_path / "config.json"
    config.write_text('{"model_type":"qwen_image"}', encoding="utf-8")

    assert ModelInspector().inspect([weights], config=config).config == {
        "model_type": "qwen_image"
    }

    config.write_bytes(b"{" + (b" " * (8 * 1024 * 1024)) + b"}")
    with pytest.raises(ModelInspectionError, match="too large"):
        ModelInspector().inspect([weights], config=config)


@pytest.mark.parametrize(
    "header",
    (
        {
            "tensor": {
                "dtype": "F16",
                "shape": [2, 2],
                "data_offsets": [0, 2],
            }
        },
        {
            "first": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            "second": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [1, 3],
            },
        },
        {
            "tensor": {
                "dtype": "MADE_UP",
                "shape": [1],
                "data_offsets": [0, 1],
            }
        },
    ),
)
def test_inspector_rejects_tensor_byte_length_overlap_and_unknown_dtype(
    tmp_path: Path,
    header: dict[str, object],
) -> None:
    from comfyng.models.inspection import ModelInspectionError, ModelInspector

    encoded = json.dumps(header, separators=(",", ":")).encode()
    data_length = max(
        entry["data_offsets"][1]  # type: ignore[index]
        for entry in header.values()
    )
    path = tmp_path / "malformed"
    path.write_bytes(
        len(encoded).to_bytes(8, "little") + encoded + bytes(data_length)
    )

    with pytest.raises(ModelInspectionError):
        ModelInspector().inspect([path])


def test_inspector_rejects_duplicate_tensor_names_across_shards(tmp_path: Path) -> None:
    from comfyng.models.inspection import ModelInspectionError, ModelInspector

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(_safetensors_bytes({"duplicate": ("F16", (1,))}))
    second.write_bytes(_safetensors_bytes({"duplicate": ("F16", (1,))}))

    with pytest.raises(ModelInspectionError, match="duplicate tensor"):
        ModelInspector().inspect([first, second])
