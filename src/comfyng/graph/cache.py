from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import msgspec

from comfyng.core.contracts import Contract

from .types import NodeInstance


CACHE_KEY_VERSION = 1


def _canonical_value(value: object) -> object:
    if isinstance(value, Contract):
        return _canonical_value(value.to_builtins())
    if isinstance(value, msgspec.Struct):
        return _canonical_value(msgspec.to_builtins(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported cache-key value: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_cache_key(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


def node_cache_key(
    node: NodeInstance,
    upstream: Mapping[str, Any] | None = None,
    *,
    cache_key_version: int = CACHE_KEY_VERSION,
) -> str:
    """Hash execution content without coupling the key to a node UUID."""

    if not isinstance(node, NodeInstance):
        raise TypeError("node must be a NodeInstance")
    if type(cache_key_version) is not int or cache_key_version <= 0:
        raise ValueError("cache_key_version must be a positive integer")
    return content_cache_key(
        {
            "cache_key_version": cache_key_version,
            "node_type": node.type_id,
            "node_version": node.type_version,
            "inputs": node.inputs,
            "metadata": node.metadata,
            "upstream": upstream or {},
        }
    )
