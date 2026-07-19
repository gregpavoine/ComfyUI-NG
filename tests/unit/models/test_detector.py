from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[int, ...]],
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    header: dict[str, object] = {}
    offset = 0
    for name, shape in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = max(elements * 2, 0)
        header[name] = {
            "dtype": "F16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(offset))
    return path


def test_detector_uses_explicit_evidence_precedence_and_reports_all_evidence(
    tmp_path: Path,
) -> None:
    from comfyng.models.detector import ArchitectureDetector
    from comfyng.models.inspection import ModelInspector

    path = _write_safetensors(
        tmp_path / "anything.random",
        {"transformer.double_blocks.0.img_attn.qkv.weight": (12, 8)},
        metadata={"comfyng.family": "qwen_image"},
    )
    inspection = ModelInspector().inspect(
        [path],
        config={"model_type": "z_image"},
        provider_declaration={"family": "krea2"},
    )
    detector = ArchitectureDetector(
        known_hashes={inspection.aggregate_sha256: "flux1"}
    )

    result = detector.detect(inspection)

    assert result.family == "flux1"
    assert result.architecture == "flux-transformer-2d"
    assert result.supported is True
    assert result.selected_source == "known_hash"
    assert [evidence.source for evidence in result.evidence] == [
        "known_hash",
        "config",
        "safetensors_metadata",
        "provider_declaration",
        "tensor_signature",
    ]


@pytest.mark.parametrize(
    ("config", "metadata", "provider", "expected", "source"),
    (
        ({"model_type": "qwen_image_edit"}, {"comfyng.family": "flux1"}, {"family": "z_image"}, "qwen_image_edit", "config"),
        ({}, {"comfyng.family": "qwen_image"}, {"family": "z_image"}, "qwen_image", "safetensors_metadata"),
        ({}, {}, {"family": "z_image"}, "z_image", "provider_declaration"),
        ({}, {}, {}, "flux1", "tensor_signature"),
    ),
)
def test_detector_falls_back_through_config_metadata_provider_then_shapes(
    tmp_path: Path,
    config: dict[str, object],
    metadata: dict[str, str],
    provider: dict[str, object],
    expected: str,
    source: str,
) -> None:
    from comfyng.models.detector import ArchitectureDetector
    from comfyng.models.inspection import ModelInspector

    path = _write_safetensors(
        tmp_path / "misleading-sd15-name.ckpt",
        {"transformer.double_blocks.0.img_attn.qkv.weight": (12, 8)},
        metadata=metadata,
    )
    inspection = ModelInspector().inspect(
        [path], config=config, provider_declaration=provider
    )

    result = ArchitectureDetector().detect(inspection)

    assert result.family == expected
    assert result.selected_source == source


@pytest.mark.parametrize(
    ("family", "tensors", "message"),
    (
        (
            "sd15",
            {
                "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
                "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": (1, 768),
            },
            "Stable Diffusion 1.5 is not supported.",
        ),
        (
            "sd2",
            {
                "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
                "cond_stage_model.model.token_embedding.weight": (1, 1024),
            },
            "Stable Diffusion 2.x is not supported.",
        ),
        (
            "sdxl",
            {
                "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
                "conditioner.embedders.1.model.text_model.embeddings.token_embedding.weight": (1, 1280),
                "model.diffusion_model.label_emb.0.0.weight": (1, 2816),
            },
            "Stable Diffusion XL is not supported.",
        ),
    ),
)
def test_renamed_legacy_weights_are_detected_and_refused_structurally(
    tmp_path: Path,
    family: str,
    tensors: dict[str, tuple[int, ...]],
    message: str,
) -> None:
    from comfyng.models.detector import ArchitectureDetector
    from comfyng.models.inspection import ModelInspector
    from comfyng.models.legacy import UnsupportedModelGeneration, require_modern

    path = _write_safetensors(tmp_path / "flux2-totally-safe-looking.bin", tensors)
    result = ArchitectureDetector().detect(ModelInspector().inspect([path]))

    assert result.family == family
    assert result.supported is False
    with pytest.raises(UnsupportedModelGeneration) as caught:
        require_modern(result)
    assert caught.value.code == "unsupported_model_generation"
    assert caught.value.minimum_generation == "FLUX.1"
    assert caught.value.detected_family == family
    assert str(caught.value) == message
    assert caught.value.to_payload() == {
        "error": {
            "code": "unsupported_model_generation",
            "message": message,
            "minimum_generation": "FLUX.1",
            "detected_family": family,
        }
    }


def test_detector_rejects_unknown_and_same_priority_conflicting_evidence(
    tmp_path: Path,
) -> None:
    from comfyng.models.detector import (
        AmbiguousArchitectureError,
        ArchitectureDetector,
        UnknownArchitectureError,
    )
    from comfyng.models.inspection import ModelInspector

    unknown_path = _write_safetensors(tmp_path / "unknown", {"other": (1,)})
    with pytest.raises(UnknownArchitectureError):
        ArchitectureDetector().detect(ModelInspector().inspect([unknown_path]))

    first = _write_safetensors(
        tmp_path / "first", {"a": (1,)}, metadata={"comfyng.family": "flux1"}
    )
    second = _write_safetensors(
        tmp_path / "second", {"b": (1,)}, metadata={"comfyng.family": "qwen_image"}
    )
    with pytest.raises(AmbiguousArchitectureError, match="safetensors_metadata"):
        ArchitectureDetector().detect(ModelInspector().inspect([first, second]))


@pytest.mark.parametrize(
    ("declared", "family", "architecture"),
    (
        ("FluxTransformer2DModel", "flux1", "flux-transformer-2d"),
        ("Flux2Transformer2DModel", "flux2", "flux2-transformer-2d"),
        ("QwenImageTransformer2DModel", "qwen_image", "qwen-image-transformer-2d"),
        (
            "QwenImageEditTransformer2DModel",
            "qwen_image_edit",
            "qwen-image-edit-transformer-2d",
        ),
        ("ZImageTransformer2DModel", "z_image", "z-image-transformer-2d"),
        ("Krea2Transformer2DModel", "krea2", "krea2-transformer-2d"),
    ),
)
def test_all_v1_modern_families_have_structured_capabilities(
    tmp_path: Path,
    declared: str,
    family: str,
    architecture: str,
) -> None:
    from comfyng.models.detector import ArchitectureDetector
    from comfyng.models.inspection import ModelInspector

    path = _write_safetensors(tmp_path / "renamed", {"neutral": (1,)})
    result = ArchitectureDetector().detect(
        ModelInspector().inspect([path], config={"architectures": [declared]})
    )

    assert result.family == family
    assert result.architecture == architecture
    assert result.supported is True
    assert result.capabilities is not None
    assert result.capabilities.family == family
    assert result.capabilities.architecture == architecture
    assert result.capabilities.task_types == frozenset(
        {"text-to-image", "image-to-image"}
    )


def test_known_hash_evidence_matches_an_individual_multifile_shard(
    tmp_path: Path,
) -> None:
    from comfyng.models.detector import ArchitectureDetector
    from comfyng.models.inspection import ModelInspector

    first = _write_safetensors(tmp_path / "first", {"neutral-a": (1,)})
    second = _write_safetensors(tmp_path / "second", {"neutral-b": (1,)})
    inspection = ModelInspector().inspect(
        [first, second], provider_declaration={"family": "z_image"}
    )

    result = ArchitectureDetector(
        known_hashes={inspection.files[1].sha256: "qwen_image"}
    ).detect(inspection)

    assert result.family == "qwen_image"
    assert result.selected_source == "known_hash"


@pytest.mark.parametrize(
    ("width", "family"),
    ((768, "sd15"), (1024, "sd2")),
)
def test_legacy_unet_cross_attention_width_is_a_structural_fallback(
    tmp_path: Path,
    width: int,
    family: str,
) -> None:
    from comfyng.models.detector import ArchitectureDetector
    from comfyng.models.inspection import ModelInspector

    path = _write_safetensors(
        tmp_path / "renamed-modern-looking",
        {
            "model.diffusion_model.input_blocks.0.0.weight": (2, 4, 3, 3),
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (
                2,
                width,
            ),
        },
    )

    result = ArchitectureDetector().detect(ModelInspector().inspect([path]))

    assert result.family == family
    assert result.supported is False
