from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


def load_settings(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    from comfyng.config import load_settings as load

    return load(path=path, env=env)


def test_defaults_are_exact_and_frozen(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data_root = home / ".local" / "share" / "comfyui-ng"

    settings = load_settings(env={"HOME": str(home)})

    assert settings.model_dump(mode="json") == {
        "server": {"host": "127.0.0.1", "port": 8188, "workers": 2},
        "runtime": {"python": ">=3.14", "multiprocessing_start": "forkserver"},
        "scheduler": {
            "default_profile": "balanced",
            "interactive_priority": 80,
            "max_queued_jobs": 100,
        },
        "cpu": {"reserve_cores": 2, "compute_workers": "auto", "io_workers": 4},
        "memory": {"reserve_system_gb": 4.0, "max_pinned_gb": 8.0},
        "gpu": {
            "devices": "auto",
            "reserve_vram_mb": 768,
            "heavy_workers_per_gpu": 1,
            "compile": "auto",
            "attention_backend": "auto",
        },
        "plugins": {
            "isolation": True,
            "lazy_loading": True,
            "default_idle_timeout": 120,
            "allow_legacy_bridge": False,
        },
        "providers": {
            "huggingface": {"enabled": True, "offline": False},
            "civitai_red": {"enabled": False},
        },
        "auth": {"mode": "NONE_LOCALHOST"},
        "data_root": str(data_root),
        "database": {
            "path": str(data_root / "comfyng.db"),
            "busy_timeout_ms": 5000,
        },
        "storage": {"root": str(data_root / "storage")},
    }

    with pytest.raises(Exception, match="frozen"):
        settings.server.port = 9000


def test_comfyng_home_wins_over_xdg_and_home(tmp_path: Path) -> None:
    settings = load_settings(
        env={
            "COMFYNG_HOME": str(tmp_path / "comfyng"),
            "XDG_DATA_HOME": str(tmp_path / "xdg"),
            "HOME": str(tmp_path / "home"),
        }
    )

    assert settings.data_root == (tmp_path / "comfyng").resolve()
    assert settings.database.path == (tmp_path / "comfyng" / "comfyng.db").resolve()
    assert settings.storage.root == (tmp_path / "comfyng" / "storage").resolve()


def test_xdg_data_home_is_used_without_comfyng_home(tmp_path: Path) -> None:
    settings = load_settings(
        env={
            "XDG_DATA_HOME": str(tmp_path / "xdg"),
            "HOME": str(tmp_path / "home"),
        }
    )

    assert settings.data_root == (tmp_path / "xdg" / "comfyui-ng").resolve()


def test_environment_overrides_yaml_and_expands_placeholders(tmp_path: Path) -> None:
    data_root = tmp_path / "configured"
    config = tmp_path / "custom.yaml"
    config.write_text(
        """
server:
  port: 9000
cpu:
  io_workers: 6
database:
  path: ${DATA_ROOT}/state/comfyng.db
storage:
  root: ${DATA_ROOT}/objects
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(
        config,
        env={
            "COMFYNG_HOME": str(data_root),
            "DATA_ROOT": str(data_root),
            "COMFYNG_SERVER__PORT": "9191",
            "COMFYNG_CPU__IO_WORKERS": "8",
            "COMFYNG_PROVIDERS__HUGGINGFACE__OFFLINE": "true",
        },
    )

    assert settings.server.port == 9191
    assert settings.cpu.io_workers == 8
    assert settings.providers.huggingface.offline is True
    assert settings.database.path == (data_root / "state" / "comfyng.db").resolve()
    assert settings.storage.root == (data_root / "objects").resolve()


def test_missing_environment_placeholder_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "missing-env.yaml"
    config.write_text("data_root: ${MISSING_ROOT}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MISSING_ROOT"):
        load_settings(config, env={})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("COMFYNG_SERVER__PORT", "0"),
        ("COMFYNG_SERVER__PORT", "65536"),
        ("COMFYNG_SERVER__WORKERS", "0"),
        ("COMFYNG_SCHEDULER__INTERACTIVE_PRIORITY", "-1"),
        ("COMFYNG_SCHEDULER__INTERACTIVE_PRIORITY", "101"),
        ("COMFYNG_SCHEDULER__MAX_QUEUED_JOBS", "0"),
        ("COMFYNG_CPU__RESERVE_CORES", "-1"),
        ("COMFYNG_CPU__COMPUTE_WORKERS", "0"),
        ("COMFYNG_CPU__IO_WORKERS", "0"),
        ("COMFYNG_MEMORY__RESERVE_SYSTEM_GB", "-1"),
        ("COMFYNG_MEMORY__MAX_PINNED_GB", "-1"),
        ("COMFYNG_GPU__RESERVE_VRAM_MB", "-1"),
        ("COMFYNG_GPU__HEAVY_WORKERS_PER_GPU", "0"),
        ("COMFYNG_PLUGINS__DEFAULT_IDLE_TIMEOUT", "0"),
        ("COMFYNG_DATABASE__BUSY_TIMEOUT_MS", "0"),
    ],
)
def test_invalid_ports_and_resource_budgets_are_rejected(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_settings(
            env={
                "HOME": str(tmp_path),
                name: value,
            }
        )


def test_paths_must_remain_under_the_data_root(tmp_path: Path) -> None:
    config = tmp_path / "unsafe.yaml"
    config.write_text(
        f"database:\n  path: {tmp_path / 'outside.db'}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data_root"):
        load_settings(config, env={"COMFYNG_HOME": str(tmp_path / "data")})


def test_unknown_configuration_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "unknown.yaml"
    config.write_text("server:\n  mystery: true\n", encoding="utf-8")

    with pytest.raises(Exception, match="mystery"):
        load_settings(config, env={"COMFYNG_HOME": str(tmp_path / "data")})


def test_settings_classmethod_uses_the_public_loader(tmp_path: Path) -> None:
    from comfyng.config import Settings

    settings = Settings.load(env={"COMFYNG_HOME": str(tmp_path)})

    assert settings.server.port == 8188
    assert settings.data_root == tmp_path.resolve()
