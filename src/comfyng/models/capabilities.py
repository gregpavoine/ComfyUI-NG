from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

from comfyng.core.contracts import Contract, register_contract
from comfyng.core.ids import validate_sha256
from comfyng.core.json_values import (
    freeze_json_value,
    validate_safe_unicode_string,
)


def _non_empty(value: object, *, field: str) -> str:
    value = validate_safe_unicode_string(value, field=field)
    if not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _unique_non_empty(
    values: object,
    *,
    field: str,
    collection_type: type[tuple] | type[frozenset],
) -> None:
    if type(values) is not collection_type or not values:
        raise ValueError(
            f"{field} must be a non-empty {collection_type.__name__}"
        )
    for item in values:
        if not validate_safe_unicode_string(item, field=field):
            raise ValueError(f"{field} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicates")


@register_contract
class ModelCapabilities(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.model-capabilities"

    family: str
    architecture: str
    task_types: frozenset[str]
    latent_channels: int
    latent_scale_factor: int
    prediction_type: str
    supported_dtypes: tuple[str, ...]
    supported_quantizations: tuple[str, ...]
    text_encoder_layout: tuple[str, ...]
    supports_negative_prompt: bool
    supports_cfg: bool
    supports_embedded_guidance: bool
    supports_img2img: bool
    supports_inpainting: bool
    supports_lora: bool
    supports_control: bool
    samplers: tuple[str, ...]
    schedulers: tuple[str, ...]
    attention_backends: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty(self.family, field="family")
        _non_empty(self.architecture, field="architecture")
        _non_empty(self.prediction_type, field="prediction_type")
        if type(self.latent_channels) is not int or self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if type(self.latent_scale_factor) is not int or self.latent_scale_factor <= 0:
            raise ValueError("latent_scale_factor must be positive")
        _unique_non_empty(
            self.task_types,
            field="task_types",
            collection_type=frozenset,
        )
        for field, values in (
            ("supported_dtypes", self.supported_dtypes),
            ("supported_quantizations", self.supported_quantizations),
            ("text_encoder_layout", self.text_encoder_layout),
            ("samplers", self.samplers),
            ("schedulers", self.schedulers),
            ("attention_backends", self.attention_backends),
        ):
            _unique_non_empty(values, field=field, collection_type=tuple)
        for field in (
            "supports_negative_prompt",
            "supports_cfg",
            "supports_embedded_guidance",
            "supports_img2img",
            "supports_inpainting",
            "supports_lora",
            "supports_control",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be a boolean")


@register_contract
class ModelHandle(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.model-handle"

    id: UUID
    family: str
    architecture: str
    local_path: Path
    sha256: str
    size_bytes: int
    source_provider: str | None
    source_model_id: str | None
    source_revision: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            raise ValueError("id must be a UUID")
        _non_empty(self.family, field="family")
        _non_empty(self.architecture, field="architecture")
        if not isinstance(self.local_path, Path) or not self.local_path.is_absolute():
            raise ValueError("local_path must be absolute")
        validate_safe_unicode_string(str(self.local_path), field="local_path")
        validate_sha256(self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_value(self.metadata, path="$.metadata"),
        )
        for field in ("source_provider", "source_model_id", "source_revision"):
            value = getattr(self, field)
            if value is not None:
                if not validate_safe_unicode_string(value, field=field):
                    raise ValueError(f"{field} must be null or a non-empty string")
