from __future__ import annotations

import re
from typing import Any

from .errors import IdentifierValidationError


_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_DOTTED_ID_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?)+$"
)
_TYPE_NAME_PATTERN = re.compile(r"^NG_[A-Z0-9]+(?:_[A-Z0-9]+)*$")
_ENTRYPOINT_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_semver(value: object, *, field: str = "version") -> str:
    if not isinstance(value, str) or _SEMVER_PATTERN.fullmatch(value) is None:
        raise IdentifierValidationError(f"{field} must be a semantic version")
    return value


def semver_sort_key(value: str) -> tuple[Any, ...]:
    match = _SEMVER_PATTERN.fullmatch(value)
    if match is None:
        raise IdentifierValidationError("version must be a semantic version")
    major, minor, patch, prerelease, _build = match.groups()
    if prerelease is None:
        prerelease_key: tuple[Any, ...] = (1, ())
    else:
        identifiers = tuple(
            (0, int(item)) if item.isdigit() else (1, item)
            for item in prerelease.split(".")
        )
        prerelease_key = (0, identifiers)
    return (int(major), int(minor), int(patch), *prerelease_key)


def validate_dotted_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DOTTED_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierValidationError(
            f"{field} must be a lowercase dotted stable identifier"
        )
    return value


def validate_node_id(value: object) -> str:
    return validate_dotted_id(value, field="node id")


def validate_package_id(value: object) -> str:
    return validate_dotted_id(value, field="package id")


def validate_type_name(value: object) -> str:
    if not isinstance(value, str) or _TYPE_NAME_PATTERN.fullmatch(value) is None:
        raise IdentifierValidationError(
            "type name must use uppercase NG_NAME syntax"
        )
    return value


def validate_entrypoint(value: object) -> str:
    if not isinstance(value, str) or _ENTRYPOINT_PATTERN.fullmatch(value) is None:
        raise IdentifierValidationError(
            "runtime entrypoint must use module.path:attribute syntax"
        )
    return value


def validate_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise IdentifierValidationError("sha256 must be 64 lowercase hexadecimal digits")
    return value


def validate_port_name(value: object, *, field: str = "port") -> str:
    if not isinstance(value, str) or _PORT_PATTERN.fullmatch(value) is None:
        raise IdentifierValidationError(f"{field} must use lower_snake_case syntax")
    return value
