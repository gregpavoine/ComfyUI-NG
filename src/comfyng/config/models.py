from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrozenModel(BaseModel):
    """Base for immutable configuration sections with closed schemas."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ServerSettings(FrozenModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8188, ge=1, le=65_535)
    workers: PositiveInt = 2


class RuntimeSettings(FrozenModel):
    python: Literal[">=3.14"] = ">=3.14"
    multiprocessing_start: Literal["forkserver", "spawn"] = "forkserver"


class SchedulerSettings(FrozenModel):
    default_profile: str = "balanced"
    interactive_priority: int = Field(default=80, ge=0, le=100)
    max_queued_jobs: PositiveInt = 100


class CpuSettings(FrozenModel):
    reserve_cores: NonNegativeInt = 2
    compute_workers: Literal["auto"] | PositiveInt = "auto"
    io_workers: PositiveInt = 4


class MemorySettings(FrozenModel):
    reserve_system_gb: NonNegativeFloat = 4
    max_pinned_gb: NonNegativeFloat = 8


class GpuSettings(FrozenModel):
    devices: str = "auto"
    reserve_vram_mb: NonNegativeInt = 768
    heavy_workers_per_gpu: PositiveInt = 1
    compile: str = "auto"
    attention_backend: str = "auto"


class PluginSettings(FrozenModel):
    isolation: bool = True
    lazy_loading: bool = True
    default_idle_timeout: PositiveInt = 120
    allow_legacy_bridge: bool = False


class HuggingFaceProviderSettings(FrozenModel):
    enabled: bool = True
    offline: bool = False


class CivitaiRedProviderSettings(FrozenModel):
    enabled: bool = False


class ProviderSettings(FrozenModel):
    huggingface: HuggingFaceProviderSettings = Field(
        default_factory=HuggingFaceProviderSettings
    )
    civitai_red: CivitaiRedProviderSettings = Field(
        default_factory=CivitaiRedProviderSettings
    )


class AuthSettings(FrozenModel):
    mode: Literal["NONE_LOCALHOST", "API_KEY", "JWT"] = "NONE_LOCALHOST"


class DatabaseSettings(FrozenModel):
    path: Path
    busy_timeout_ms: PositiveInt = 5_000


class StorageSettings(FrozenModel):
    root: Path


def _resolved_path(value: str | Path, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=False)


class Settings(BaseSettings):
    """Complete immutable configuration for the ComfyUI-NG control plane."""

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        env_prefix="COMFYNG_",
        env_nested_delimiter="__",
        validate_default=True,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    cpu: CpuSettings = Field(default_factory=CpuSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    gpu: GpuSettings = Field(default_factory=GpuSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    data_root: Path
    database: DatabaseSettings
    storage: StorageSettings

    @model_validator(mode="before")
    @classmethod
    def resolve_and_derive_paths(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        payload = dict(value)
        if "data_root" not in payload:
            raise ValueError("data_root must be resolved before validating settings")

        data_root = _resolved_path(payload["data_root"], relative_to=Path.cwd())
        payload["data_root"] = data_root

        database_value = payload.get("database") or {}
        database = (
            database_value.model_dump()
            if isinstance(database_value, BaseModel)
            else dict(database_value)
        )
        database["path"] = _resolved_path(
            database.get("path", data_root / "comfyng.db"),
            relative_to=data_root,
        )
        payload["database"] = database

        storage_value = payload.get("storage") or {}
        storage = (
            storage_value.model_dump()
            if isinstance(storage_value, BaseModel)
            else dict(storage_value)
        )
        storage["root"] = _resolved_path(
            storage.get("root", data_root / "storage"),
            relative_to=data_root,
        )
        payload["storage"] = storage
        return payload

    @model_validator(mode="after")
    def keep_paths_under_data_root(self) -> Self:
        for name, path in (
            ("database.path", self.database.path),
            ("storage.root", self.storage.root),
        ):
            if not path.is_relative_to(self.data_root):
                raise ValueError(f"{name} must remain under data_root")
        return self

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Self:
        from .loader import load_settings

        return load_settings(path=path, env=env)
