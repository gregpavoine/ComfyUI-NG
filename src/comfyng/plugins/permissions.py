from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import re


_PERMISSION_FIELDS = frozenset(
    (
        "network",
        "filesystem_read",
        "filesystem_write",
        "subprocess",
        "gpu",
        "camera",
        "microphone",
    )
)
_ROOT_ALIAS = re.compile(r"^[a-z][a-z0-9_]*$")


class PermissionDenied(PermissionError):
    def __init__(self, permission: str, target: Path | str | None = None) -> None:
        self.permission = permission
        self.target = target
        detail = "" if target is None else f" for {target}"
        super().__init__(f"plugin permission {permission!r} denied{detail}")


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"permission {name} must be a boolean")
    return value


def _aliases(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"permission {name} must be an array of root aliases")
    aliases: list[str] = []
    for alias in value:
        if not isinstance(alias, str) or _ROOT_ALIAS.fullmatch(alias) is None:
            raise ValueError(f"permission {name} contains an invalid root alias")
        aliases.append(alias)
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"permission {name} contains duplicate root aliases")
    return tuple(sorted(aliases))


@dataclass(frozen=True, slots=True)
class PermissionSet:
    network: bool = False
    filesystem_read: tuple[str, ...] = ()
    filesystem_write: tuple[str, ...] = ()
    subprocess: bool = False
    gpu: bool = False
    camera: bool = False
    microphone: bool = False

    def __post_init__(self) -> None:
        for name in ("network", "subprocess", "gpu", "camera", "microphone"):
            _boolean(name, getattr(self, name))
        for name in ("filesystem_read", "filesystem_write"):
            value = getattr(self, name)
            normalized = _aliases(name, value)
            object.__setattr__(self, name, normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> PermissionSet:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("permissions must be a mapping")
        unknown = set(value) - _PERMISSION_FIELDS
        if unknown:
            raise ValueError(
                f"unknown plugin permissions: {', '.join(sorted(map(str, unknown)))}"
            )
        return cls(
            network=_boolean("network", value.get("network", False)),
            filesystem_read=_aliases(
                "filesystem_read", value.get("filesystem_read", ())
            ),
            filesystem_write=_aliases(
                "filesystem_write", value.get("filesystem_write", ())
            ),
            subprocess=_boolean("subprocess", value.get("subprocess", False)),
            gpu=_boolean("gpu", value.get("gpu", False)),
            camera=_boolean("camera", value.get("camera", False)),
            microphone=_boolean("microphone", value.get("microphone", False)),
        )

    def to_mapping(self) -> dict[str, bool | tuple[str, ...]]:
        return {
            "network": self.network,
            "filesystem_read": self.filesystem_read,
            "filesystem_write": self.filesystem_write,
            "subprocess": self.subprocess,
            "gpu": self.gpu,
            "camera": self.camera,
            "microphone": self.microphone,
        }


class PermissionGuard:
    """Path-aware permission checks used before dispatching privileged work."""

    def __init__(
        self,
        permissions: PermissionSet,
        roots: Mapping[str, Path | str],
    ) -> None:
        if not isinstance(permissions, PermissionSet):
            raise TypeError("permissions must be a PermissionSet")
        resolved: dict[str, Path] = {}
        for alias, raw_path in roots.items():
            if not isinstance(alias, str) or _ROOT_ALIAS.fullmatch(alias) is None:
                raise ValueError("permission root aliases are invalid")
            path = Path(raw_path).resolve(strict=True)
            if not path.is_dir():
                raise ValueError(f"permission root {alias!r} must be a directory")
            resolved[alias] = path
        declared = set(permissions.filesystem_read).union(permissions.filesystem_write)
        missing = declared - set(resolved)
        if missing:
            raise ValueError(
                f"permission roots are missing: {', '.join(sorted(missing))}"
            )
        self.permissions = permissions
        self.roots = MappingProxyType(resolved)

    def _require_capability(self, name: str) -> None:
        if not getattr(self.permissions, name):
            raise PermissionDenied(name)

    def require_network(self) -> None:
        self._require_capability("network")

    def require_subprocess(self) -> None:
        self._require_capability("subprocess")

    def require_gpu(self) -> None:
        self._require_capability("gpu")

    def require_camera(self) -> None:
        self._require_capability("camera")

    def require_microphone(self) -> None:
        self._require_capability("microphone")

    def _require_path(
        self,
        value: Path | str,
        *,
        permission: str,
        aliases: tuple[str, ...],
        strict: bool,
    ) -> Path:
        raw_path = Path(value)
        try:
            path = raw_path.resolve(strict=strict)
        except (OSError, RuntimeError) as exc:
            raise PermissionDenied(permission, raw_path) from exc
        for alias in aliases:
            if path.is_relative_to(self.roots[alias]):
                return path
        raise PermissionDenied(permission, raw_path)

    def require_read(self, value: Path | str) -> Path:
        return self._require_path(
            value,
            permission="filesystem_read",
            aliases=self.permissions.filesystem_read,
            strict=True,
        )

    def require_write(self, value: Path | str) -> Path:
        return self._require_path(
            value,
            permission="filesystem_write",
            aliases=self.permissions.filesystem_write,
            strict=False,
        )


__all__ = ["PermissionDenied", "PermissionGuard", "PermissionSet"]
