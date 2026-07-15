from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from .errors import JsonValueValidationError


_PATH_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MIN_JSON_INTEGER = -(2**63)
_MAX_JSON_INTEGER = 2**63 - 1


class FrozenDict(dict[str, Any]):
    """A dict-compatible, recursively immutable contract mapping."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def copy(self) -> FrozenDict:
        return self


def validate_safe_unicode_string(value: object, *, field: str) -> str:
    """Validate a public string before it reaches a contract encoder."""

    if type(value) is not str:
        raise ValueError(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{field} must contain valid Unicode scalar values"
        ) from exc
    return value


def _child_path(path: str, key: str) -> str:
    if _PATH_NAME.fullmatch(key):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def validate_json_value(value: object, *, path: str = "$") -> None:
    """Reject values that cannot round-trip through strict JSON unchanged."""

    freeze_json_value(value, path=path)


def freeze_json_value(value: object, *, path: str = "$") -> object:
    """Deep-copy a JSON value into canonical immutable contract containers."""

    return _freeze_json_value(value, path=path, active_containers=set())


def _freeze_json_value(
    value: object,
    *,
    path: str,
    active_containers: set[int],
) -> object:
    value_type = type(value)
    if value is None or value_type is bool:
        return value
    if value_type is str:
        try:
            value.encode("utf-8")  # type: ignore[union-attr]
        except UnicodeEncodeError as exc:
            raise JsonValueValidationError(
                path, "string must contain valid Unicode scalar values"
            ) from exc
        return value
    if value_type is int:
        if not _MIN_JSON_INTEGER <= value <= _MAX_JSON_INTEGER:  # type: ignore[operator]
            raise JsonValueValidationError(path, "integer is outside signed 64-bit range")
        return value
    if value_type is float:
        if not math.isfinite(value):  # type: ignore[arg-type]
            raise JsonValueValidationError(path, "float must be finite")
        return value
    if value_type is list or value_type is tuple:
        identity = id(value)
        if identity in active_containers:
            raise JsonValueValidationError(path, "cyclic arrays are not JSON-compatible")
        active_containers.add(identity)
        try:
            return tuple(
                _freeze_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)  # type: ignore[arg-type]
            )
        finally:
            active_containers.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise JsonValueValidationError(path, "cyclic objects are not JSON-compatible")
        active_containers.add(identity)
        try:
            frozen_items: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise JsonValueValidationError(
                        path, "JSON object keys must be strings"
                    )
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise JsonValueValidationError(
                        path, "JSON object keys must contain valid Unicode"
                    ) from exc
                frozen_items[key] = _freeze_json_value(
                    item,
                    path=_child_path(path, key),
                    active_containers=active_containers,
                )
            return FrozenDict(frozen_items)
        finally:
            active_containers.remove(identity)
    raise JsonValueValidationError(
        path,
        f"{value_type.__name__} is not a stable JSON value",
    )
