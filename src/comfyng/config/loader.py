from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib.resources import files
import os
from pathlib import Path
import re
from typing import Any

import yaml

from .models import Settings


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_PREFIX = "COMFYNG_"
_SPECIAL_ENV_KEYS = {"COMFYNG_HOME"}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_environment(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in env:
                raise ValueError(f"missing environment variable: {name}")
            return env[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item, env) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _expand_environment(item, env)
            for key, item in value.items()
        }
    return value


def _parse_yaml(text: str, *, source: str, env: Mapping[str, str]) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"configuration root in {source} must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"configuration keys in {source} must be strings")
    return dict(_expand_environment(loaded, env))


def _load_default_config(env: Mapping[str, str]) -> dict[str, Any]:
    source_path = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
    if source_path.is_file():
        return _parse_yaml(
            source_path.read_text(encoding="utf-8"),
            source=str(source_path),
            env=env,
        )

    resource = files("comfyng.config").joinpath("default.yaml")
    return _parse_yaml(resource.read_text(encoding="utf-8"), source=str(resource), env=env)


def _load_override(path: Path, env: Mapping[str, str]) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"configuration path is not a file: {resolved}")
    return _parse_yaml(
        resolved.read_text(encoding="utf-8"),
        source=str(resolved),
        env=env,
    )


def _assign_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def _apply_environment(payload: dict[str, Any], env: Mapping[str, str]) -> None:
    for name, raw_value in env.items():
        if name in _SPECIAL_ENV_KEYS or not name.startswith(_ENV_PREFIX):
            continue
        suffix = name.removeprefix(_ENV_PREFIX)
        if not suffix:
            continue
        keys = [part.lower() for part in suffix.split("__")]
        _assign_nested(payload, keys, yaml.safe_load(raw_value))


def _default_data_root(env: Mapping[str, str]) -> Path:
    if comfyng_home := env.get("COMFYNG_HOME"):
        return Path(comfyng_home).expanduser()
    if xdg_data_home := env.get("XDG_DATA_HOME"):
        return Path(xdg_data_home).expanduser() / "comfyui-ng"
    home = Path(env.get("HOME", str(Path.home()))).expanduser()
    return home / ".local" / "share" / "comfyui-ng"


def load_settings(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load defaults, an optional YAML overlay, then environment overrides."""

    environment = dict(os.environ if env is None else env)
    payload = _load_default_config(environment)
    if path is not None:
        payload = _deep_merge(payload, _load_override(path, environment))

    _apply_environment(payload, environment)
    if "COMFYNG_DATA_ROOT" not in environment and "COMFYNG_HOME" in environment:
        payload["data_root"] = environment["COMFYNG_HOME"]
    payload.setdefault("data_root", _default_data_root(environment))
    return Settings.model_validate(payload)
