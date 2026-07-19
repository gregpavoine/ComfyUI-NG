from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from .capabilities import ModelCapabilities
from .inspection import ModelInspection


class ArchitectureDetectionError(ValueError):
    """Base class for deterministic model architecture detection failures."""


class UnknownArchitectureError(ArchitectureDetectionError):
    """Raised when the supplied evidence does not identify a supported format."""


class AmbiguousArchitectureError(ArchitectureDetectionError):
    """Raised when evidence at one trust level identifies multiple families."""


@dataclass(frozen=True, slots=True)
class ArchitectureEvidence:
    source: str
    family: str
    architecture: str
    score: int
    detail: str


@dataclass(frozen=True, slots=True)
class ArchitectureDetection:
    family: str
    architecture: str
    supported: bool
    generation: str
    confidence: float
    selected_source: str
    quantization: str
    evidence: tuple[ArchitectureEvidence, ...]
    capabilities: ModelCapabilities | None


@dataclass(frozen=True, slots=True)
class _Family:
    family: str
    architecture: str
    supported: bool


_FAMILIES = {
    "flux1": _Family("flux1", "flux-transformer-2d", True),
    "flux2": _Family("flux2", "flux2-transformer-2d", True),
    "qwen_image": _Family("qwen_image", "qwen-image-transformer-2d", True),
    "qwen_image_edit": _Family(
        "qwen_image_edit", "qwen-image-edit-transformer-2d", True
    ),
    "z_image": _Family("z_image", "z-image-transformer-2d", True),
    "krea2": _Family("krea2", "krea2-transformer-2d", True),
    "sd15": _Family("sd15", "stable-diffusion-v1", False),
    "sd2": _Family("sd2", "stable-diffusion-v2", False),
    "sdxl": _Family("sdxl", "stable-diffusion-xl", False),
}


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_ALIASES: dict[str, str] = {}
for family, aliases in {
    "flux1": (
        "flux",
        "flux1",
        "flux.1",
        "flux-dev",
        "flux-schnell",
        "flux-krea",
        "flux-fill",
        "FluxTransformer2DModel",
        "FluxPipeline",
        "FluxImg2ImgPipeline",
        "FluxFillPipeline",
        "flux-transformer-2d",
    ),
    "flux2": ("flux2", "flux.2", "Flux2Transformer2DModel", "flux2-transformer-2d"),
    "qwen_image": (
        "qwen_image",
        "qwen-image",
        "QwenImageTransformer2DModel",
        "QwenImagePipeline",
        "qwen-image-transformer-2d",
    ),
    "qwen_image_edit": (
        "qwen_image_edit",
        "qwen-image-edit",
        "QwenImageEditTransformer2DModel",
        "QwenImageEditPipeline",
        "qwen-image-edit-transformer-2d",
    ),
    "z_image": ("z_image", "z-image", "ZImageTransformer2DModel"),
    "krea2": ("krea2", "krea-2", "Krea2Transformer2DModel"),
    "sd15": (
        "sd15",
        "sd1.5",
        "stable-diffusion-v1-5",
        "stable-diffusion-v1",
        "stable diffusion 1.5",
        "v1-inference",
    ),
    "sd2": (
        "sd2",
        "sd2.1",
        "stable-diffusion-2",
        "stable-diffusion-v2",
        "stable diffusion 2",
    ),
    "sdxl": (
        "sdxl",
        "sdxl_base_v1-0",
        "stable-diffusion-xl",
        "stable-diffusion-xl-v1-base",
        "StableDiffusionXLPipeline",
    ),
}.items():
    for alias in aliases:
        _ALIASES[_normal(alias)] = family


_DECLARATION_KEYS = (
    "family",
    "architecture",
    "model_type",
    "_class_name",
    "base_model",
    "base_model_type",
    "modelspec.architecture",
    "comfyng.family",
    "model_family",
    "ss_base_model_version",
)


def _family_for_value(value: Any) -> _Family | None:
    if not isinstance(value, str):
        return None
    family = _ALIASES.get(_normal(value))
    return None if family is None else _FAMILIES[family]


def _mapping_families(mapping: Mapping[str, Any]) -> dict[str, set[str]]:
    matches: dict[str, set[str]] = {}
    for key in _DECLARATION_KEYS:
        raw = mapping.get(key)
        values = raw if isinstance(raw, (list, tuple)) else (raw,)
        for value in values:
            family = _family_for_value(value)
            if family is not None:
                matches.setdefault(family.family, set()).add(f"{key}={value}")
    architectures = mapping.get("architectures")
    if isinstance(architectures, (list, tuple)):
        for value in architectures:
            family = _family_for_value(value)
            if family is not None:
                matches.setdefault(family.family, set()).add(f"architectures={value}")
    return matches


def _capabilities(family: _Family) -> ModelCapabilities | None:
    if not family.supported:
        return None
    layouts = {
        "flux1": ("clip_l", "t5xxl"),
        "flux2": ("clip_l", "t5xxl"),
        "qwen_image": ("qwen2_5_vl",),
        "qwen_image_edit": ("qwen2_5_vl",),
        "z_image": ("qwen3",),
        "krea2": ("clip_l", "t5xxl"),
    }
    embedded = family.family.startswith("flux") or family.family == "krea2"
    return ModelCapabilities(
        family=family.family,
        architecture=family.architecture,
        task_types=frozenset(("text-to-image", "image-to-image")),
        latent_channels=16,
        latent_scale_factor=8,
        prediction_type="flow_matching",
        supported_dtypes=("bfloat16", "float16", "float32"),
        supported_quantizations=("none", "int8", "nf4", "fp8"),
        text_encoder_layout=layouts[family.family],
        supports_negative_prompt=not embedded,
        supports_cfg=not embedded,
        supports_embedded_guidance=embedded,
        supports_img2img=True,
        supports_inpainting=family.family in {"flux1", "qwen_image_edit"},
        supports_lora=True,
        supports_control=True,
        samplers=("euler", "heun", "dpmpp_2m"),
        schedulers=("simple", "beta", "sgm_uniform"),
        attention_backends=("sdpa", "flash_attention_2", "xformers"),
    )


def _tensor_families(inspection: ModelInspection) -> dict[str, str]:
    names = tuple(inspection.tensor_shapes)
    matches: dict[str, str] = {}
    if any("double_blocks." in name or "single_blocks." in name for name in names):
        matches["flux1"] = "MMDiT double/single block tensor signature"
    if any("qwen" in name.casefold() for name in names) and any(
        "transformer_blocks." in name for name in names
    ):
        matches["qwen_image"] = "Qwen transformer block tensor signature"
    if any("z_image" in name.casefold() or "context_refiner" in name for name in names):
        matches["z_image"] = "Z-Image tensor signature"
    if any("krea2" in name.casefold() for name in names):
        matches["krea2"] = "KREA 2 tensor signature"

    if any("conditioner.embedders.1." in name for name in names) or any(
        "label_emb." in name for name in names
    ):
        matches["sdxl"] = "dual text encoder or SDXL label embedding signature"
    elif any("model.diffusion_model.input_blocks." in name for name in names):
        embedding_widths = {
            shape[-1]
            for name, shape in inspection.tensor_shapes.items()
            if (
                "token_embedding" in name
                or name.endswith("attn2.to_k.weight")
                or name.endswith("attn2.to_v.weight")
            )
            and shape
        }
        if 768 in embedding_widths:
            matches["sd15"] = "UNet plus 768-wide CLIP tensor signature"
        if 1024 in embedding_widths:
            matches["sd2"] = "UNet plus 1024-wide OpenCLIP tensor signature"
    return matches


def _quantization(inspection: ModelInspection) -> str:
    for mapping in (
        inspection.config,
        inspection.safetensors_metadata,
        inspection.provider_declaration,
    ):
        value = mapping.get("quantization") or mapping.get("quantization_method")
        if isinstance(value, str) and value.strip():
            return value.casefold()
        config = mapping.get("quantization_config")
        if isinstance(config, Mapping):
            method = config.get("quant_method")
            if isinstance(method, str) and method.strip():
                return method.casefold()
    dtypes = set(inspection.tensor_dtypes.values())
    if dtypes and dtypes <= {"I8", "U8"}:
        return "int8"
    return "none"


class ArchitectureDetector:
    """Select architecture from explicit byte/config evidence, never filenames."""

    def __init__(
        self,
        *,
        known_hashes: Mapping[str, str | Mapping[str, Any]] | None = None,
    ) -> None:
        self.known_hashes = {} if known_hashes is None else dict(known_hashes)

    @staticmethod
    def _append_mapping_evidence(
        evidence: list[ArchitectureEvidence],
        *,
        source: str,
        score: int,
        mappings: tuple[Mapping[str, Any], ...],
    ) -> None:
        matches: dict[str, set[str]] = {}
        for mapping in mappings:
            for family, details in _mapping_families(mapping).items():
                matches.setdefault(family, set()).update(details)
        if len(matches) > 1:
            raise AmbiguousArchitectureError(
                f"conflicting {source} evidence: {', '.join(sorted(matches))}"
            )
        for family, details in matches.items():
            definition = _FAMILIES[family]
            evidence.append(
                ArchitectureEvidence(
                    source=source,
                    family=family,
                    architecture=definition.architecture,
                    score=score,
                    detail="; ".join(sorted(details)),
                )
            )

    def detect(self, inspection: ModelInspection) -> ArchitectureDetection:
        evidence: list[ArchitectureEvidence] = []
        known_mappings: list[Mapping[str, Any]] = []
        for digest in (
            inspection.aggregate_sha256,
            *(item.sha256 for item in inspection.files),
        ):
            known = self.known_hashes.get(digest)
            if known is not None:
                known_mappings.append(
                    {"family": known} if isinstance(known, str) else known
                )
        if known_mappings:
            self._append_mapping_evidence(
                evidence,
                source="known_hash",
                score=100,
                mappings=tuple(known_mappings),
            )
        self._append_mapping_evidence(
            evidence,
            source="config",
            score=90,
            mappings=(inspection.config,),
        )
        self._append_mapping_evidence(
            evidence,
            source="repository_manifest",
            score=85,
            mappings=(inspection.repository_manifest,),
        )
        self._append_mapping_evidence(
            evidence,
            source="safetensors_metadata",
            score=80,
            mappings=tuple(item.metadata for item in inspection.files),
        )
        self._append_mapping_evidence(
            evidence,
            source="provider_declaration",
            score=70,
            mappings=(inspection.provider_declaration,),
        )
        tensor_matches = _tensor_families(inspection)
        if len(tensor_matches) > 1:
            raise AmbiguousArchitectureError(
                "conflicting tensor_signature evidence: "
                + ", ".join(sorted(tensor_matches))
            )
        for family, detail in tensor_matches.items():
            definition = _FAMILIES[family]
            evidence.append(
                ArchitectureEvidence(
                    source="tensor_signature",
                    family=family,
                    architecture=definition.architecture,
                    score=60,
                    detail=detail,
                )
            )
        if not evidence:
            raise UnknownArchitectureError(
                "model architecture is unknown; filename evidence is intentionally ignored"
            )
        ordered = tuple(
            sorted(evidence, key=lambda item: (-item.score, item.source, item.family))
        )
        selected = ordered[0]
        family = _FAMILIES[selected.family]
        return ArchitectureDetection(
            family=family.family,
            architecture=family.architecture,
            supported=family.supported,
            generation="modern" if family.supported else "legacy",
            confidence=selected.score / 100,
            selected_source=selected.source,
            quantization=_quantization(inspection),
            evidence=ordered,
            capabilities=_capabilities(family),
        )


__all__ = [
    "AmbiguousArchitectureError",
    "ArchitectureDetection",
    "ArchitectureDetectionError",
    "ArchitectureDetector",
    "ArchitectureEvidence",
    "UnknownArchitectureError",
]
