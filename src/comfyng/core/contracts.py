from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar

import msgspec

from .errors import ContractValidationError


ContractT = TypeVar("ContractT", bound="Contract")
_CONTRACT_REGISTRY: dict[tuple[str, int], type[Contract]] = {}


def _encode_hook(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def _decode_hook(expected_type: type, value: object) -> object:
    if expected_type is Path and isinstance(value, str):
        return Path(value)
    return value


class Contract(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Frozen msgspec contract with an explicit type and schema-version envelope."""

    TYPE_ID: ClassVar[str] = "comfyng.contract"
    CONTRACT_VERSION: ClassVar[int] = 1

    @property
    def contract_type(self) -> str:
        return self.TYPE_ID

    @property
    def contract_version(self) -> int:
        return self.CONTRACT_VERSION

    def to_builtins(self) -> dict[str, Any]:
        value = msgspec.to_builtins(self, enc_hook=_encode_hook)
        if not isinstance(value, dict):
            raise ContractValidationError("contract payload must encode as an object")
        return value

    @classmethod
    def from_builtins(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise ContractValidationError("contract payload must be an object")
        try:
            decoded = msgspec.convert(
                dict(value),
                type=cls,
                strict=True,
                dec_hook=_decode_hook,
            )
        except (TypeError, ValueError, msgspec.ValidationError) as exc:
            raise ContractValidationError(str(exc)) from exc
        return decoded

    def to_json(self) -> bytes:
        envelope = {
            "type": self.contract_type,
            "version": self.contract_version,
            "payload": self.to_builtins(),
        }
        return msgspec.json.encode(envelope)

    @classmethod
    def from_json(cls, value: bytes | bytearray | memoryview | str) -> Self:
        decoded = decode_contract(value, expected_type=cls)
        return decoded


def register_contract(contract_type: type[ContractT]) -> type[ContractT]:
    key = (contract_type.TYPE_ID, contract_type.CONTRACT_VERSION)
    previous = _CONTRACT_REGISTRY.get(key)
    if previous is not None and previous is not contract_type:
        raise RuntimeError(f"duplicate contract registration for {key!r}")
    _CONTRACT_REGISTRY[key] = contract_type
    return contract_type


def decode_contract(
    value: bytes | bytearray | memoryview | str,
    *,
    expected_type: type[ContractT] | None = None,
) -> ContractT:
    try:
        envelope = msgspec.json.decode(value)
    except (TypeError, ValueError, msgspec.DecodeError) as exc:
        raise ContractValidationError(f"invalid contract JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ContractValidationError("contract envelope must be an object")
    if set(envelope) != {"type", "version", "payload"}:
        raise ContractValidationError(
            "contract envelope must contain only type, version and payload"
        )
    type_id = envelope["type"]
    version = envelope["version"]
    if not isinstance(type_id, str) or type(version) is not int:
        raise ContractValidationError("contract type/version metadata is invalid")
    resolved = _CONTRACT_REGISTRY.get((type_id, version))
    if resolved is None:
        raise ContractValidationError(
            f"unsupported contract type/version: {type_id!r} v{version!r}"
        )
    if expected_type is not None and resolved is not expected_type:
        raise ContractValidationError(
            f"expected {expected_type.TYPE_ID!r}, received {type_id!r}"
        )
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise ContractValidationError("contract payload must be an object")
    return resolved.from_builtins(payload)  # type: ignore[return-value]
