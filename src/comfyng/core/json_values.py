from __future__ import annotations

import json
import math
import re

from .errors import JsonValueValidationError


_PATH_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MIN_JSON_INTEGER = -(2**63)
_MAX_JSON_INTEGER = 2**63 - 1


def _child_path(path: str, key: str) -> str:
    if _PATH_NAME.fullmatch(key):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def validate_json_value(value: object, *, path: str = "$") -> None:
    """Reject values that cannot round-trip through strict JSON unchanged."""

    _validate_json_value(value, path=path, active_containers=set())


def _validate_json_value(
    value: object,
    *,
    path: str,
    active_containers: set[int],
) -> None:
    value_type = type(value)
    if value is None or value_type is bool:
        return
    if value_type is str:
        try:
            value.encode("utf-8")  # type: ignore[union-attr]
        except UnicodeEncodeError as exc:
            raise JsonValueValidationError(
                path, "string must contain valid Unicode scalar values"
            ) from exc
        return
    if value_type is int:
        if not _MIN_JSON_INTEGER <= value <= _MAX_JSON_INTEGER:  # type: ignore[operator]
            raise JsonValueValidationError(path, "integer is outside signed 64-bit range")
        return
    if value_type is float:
        if not math.isfinite(value):  # type: ignore[arg-type]
            raise JsonValueValidationError(path, "float must be finite")
        return
    if value_type is list:
        identity = id(value)
        if identity in active_containers:
            raise JsonValueValidationError(path, "cyclic arrays are not JSON-compatible")
        active_containers.add(identity)
        try:
            for index, item in enumerate(value):  # type: ignore[union-attr]
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return
    if value_type is dict:
        identity = id(value)
        if identity in active_containers:
            raise JsonValueValidationError(path, "cyclic objects are not JSON-compatible")
        active_containers.add(identity)
        try:
            for key, item in value.items():  # type: ignore[union-attr]
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
                _validate_json_value(
                    item,
                    path=_child_path(path, key),
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return
    raise JsonValueValidationError(
        path,
        f"{value_type.__name__} is not a stable JSON value",
    )
