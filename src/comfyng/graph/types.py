from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar
from uuid import UUID

import msgspec

from comfyng.core.contracts import Contract, register_contract
from comfyng.core.enums import SerializationStrategy, TransferPolicy
from comfyng.core.errors import (
    DuplicateTypeDefinitionError,
    UnknownTypeDefinitionError,
)
from comfyng.core.ids import (
    validate_node_id,
    validate_port_name,
    validate_semver,
    validate_type_name,
)
from comfyng.core.json_values import (
    FrozenDict,
    freeze_json_value,
    validate_safe_unicode_string,
)


@register_contract
class TypeRef(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.type-ref"

    name: str
    version: int

    def __post_init__(self) -> None:
        validate_type_name(self.name)
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("type version must be a positive integer")

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, value: str) -> TypeRef:
        if not isinstance(value, str) or value.count("@") != 1:
            raise ValueError("type reference must use NG_NAME@VERSION syntax")
        name, raw_version = value.rsplit("@", 1)
        if not raw_version.isascii() or not raw_version.isdecimal():
            raise ValueError("type reference version must be a positive integer")
        return cls(name=name, version=int(raw_version))


@register_contract
class PortTypeDefinition(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.port-type-definition"

    ref: TypeRef
    schema: Mapping[str, Any]
    serialization_strategy: SerializationStrategy
    transfer_policy: TransferPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.ref, TypeRef):
            raise ValueError("ref must be a TypeRef")
        if not isinstance(self.schema, Mapping):
            raise ValueError("schema must be a JSON object")
        object.__setattr__(
            self,
            "schema",
            freeze_json_value(self.schema, path="$.schema"),
        )
        if not isinstance(self.serialization_strategy, SerializationStrategy):
            raise ValueError("serialization_strategy must be a SerializationStrategy")
        if not isinstance(self.transfer_policy, TransferPolicy):
            raise ValueError("transfer_policy must be a TransferPolicy")


class TypeRegistry:
    """Registry keyed by stable type name and positive schema version."""

    def __init__(self, definitions: Iterable[PortTypeDefinition] = ()) -> None:
        self._definitions: dict[tuple[str, int], PortTypeDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: PortTypeDefinition) -> None:
        if not isinstance(definition, PortTypeDefinition):
            raise TypeError("definition must be a PortTypeDefinition")
        key = (definition.ref.name, definition.ref.version)
        if key in self._definitions:
            raise DuplicateTypeDefinitionError(
                f"type {definition.ref} is already registered"
            )
        self._definitions[key] = definition

    def resolve(self, name: str, version: int) -> PortTypeDefinition:
        try:
            return self._definitions[(name, version)]
        except KeyError as exc:
            raise UnknownTypeDefinitionError(
                f"unknown type definition {name}@{version}"
            ) from exc

    def resolve_ref(self, value: TypeRef | str) -> PortTypeDefinition:
        ref = TypeRef.parse(value) if isinstance(value, str) else value
        return self.resolve(ref.name, ref.version)

    @property
    def definitions(self) -> tuple[PortTypeDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))


@register_contract
class TensorHandle(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.tensor-handle"

    id: UUID
    storage: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    owner_worker: str
    byte_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            raise ValueError("id must be a UUID")
        for field in ("storage", "dtype", "device", "owner_worker"):
            value = getattr(self, field)
            if not validate_safe_unicode_string(value, field=field):
                raise ValueError(f"{field} must be a non-empty string")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("shape must be a non-empty tuple")
        if any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("shape dimensions must be positive integers")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")


@register_contract
class NodeInstance(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.node-instance"

    id: UUID
    type_id: str
    type_version: str
    inputs: Mapping[str, Any] = msgspec.field(default_factory=dict)
    metadata: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            raise ValueError("id must be a UUID")
        validate_node_id(self.type_id)
        validate_semver(self.type_version, field="type_version")
        for field in ("inputs", "metadata"):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field} must be a JSON object")
            frozen = freeze_json_value(value, path=f"$.{field}")
            object.__setattr__(self, field, frozen)
            if field == "inputs":
                for key in frozen:  # type: ignore[union-attr]
                    validate_port_name(key, field="input name")


@register_contract
class Edge(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.edge"

    source_node_id: UUID
    source_port: str
    target_node_id: UUID
    target_port: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_node_id, UUID):
            raise ValueError("source_node_id must be a UUID")
        if not isinstance(self.target_node_id, UUID):
            raise ValueError("target_node_id must be a UUID")
        validate_port_name(self.source_port, field="source_port")
        validate_port_name(self.target_port, field="target_port")

    @property
    def source_node(self) -> UUID:
        return self.source_node_id

    @property
    def target_node(self) -> UUID:
        return self.target_node_id


@register_contract
class InputBinding(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.input-binding"

    node_id: UUID
    port: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, UUID):
            raise ValueError("node_id must be a UUID")
        validate_port_name(self.port, field="input binding port")


@register_contract
class OutputBinding(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.output-binding"

    node_id: UUID
    port: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, UUID):
            raise ValueError("node_id must be a UUID")
        validate_port_name(self.port, field="output binding port")


@register_contract
class Graph(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.graph"

    id: UUID
    version: int
    nodes: tuple[NodeInstance, ...]
    edges: tuple[Edge, ...]
    inputs: Mapping[str, InputBinding] = msgspec.field(default_factory=dict)
    outputs: Mapping[str, OutputBinding] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            raise ValueError("id must be a UUID")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("graph version must be a positive integer")
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(node, NodeInstance) for node in self.nodes
        ):
            raise ValueError("nodes must be a tuple of NodeInstance values")
        if not isinstance(self.edges, tuple) or any(
            not isinstance(edge, Edge) for edge in self.edges
        ):
            raise ValueError("edges must be a tuple of Edge values")
        for field, binding_type in (
            ("inputs", InputBinding),
            ("outputs", OutputBinding),
        ):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field} must be a JSON object")
            frozen: dict[str, InputBinding | OutputBinding] = {}
            for key, binding in value.items():
                validate_port_name(key, field=f"graph {field} name")
                if not isinstance(binding, binding_type):
                    raise ValueError(
                        f"graph {field} values must be {binding_type.__name__} values"
                    )
                frozen[key] = binding
            object.__setattr__(self, field, FrozenDict(frozen))


def _default_types() -> tuple[PortTypeDefinition, ...]:
    policies = {
        "NG_MODEL": TransferPolicy.HANDLE,
        "NG_MODEL_INFO": TransferPolicy.INLINE,
        "NG_TEXT_ENCODER": TransferPolicy.HANDLE,
        "NG_VAE": TransferPolicy.HANDLE,
        "NG_CONDITIONING": TransferPolicy.HANDLE,
        "NG_LATENT": TransferPolicy.HANDLE,
        "NG_IMAGE": TransferPolicy.HANDLE,
        "NG_MASK": TransferPolicy.HANDLE,
        "NG_LORA_STACK": TransferPolicy.INLINE,
        "NG_SAMPLER_CONFIG": TransferPolicy.INLINE,
        "NG_ARTIFACT": TransferPolicy.INLINE,
        "NG_JOB_REFERENCE": TransferPolicy.INLINE,
    }
    return tuple(
        PortTypeDefinition(
            ref=TypeRef(name=name, version=1),
            schema={"type": "string", "x-comfyng-type": f"{name}@1"},
            serialization_strategy=(
                SerializationStrategy.SHARED_HANDLE
                if policy is TransferPolicy.HANDLE
                else SerializationStrategy.JSON
            ),
            transfer_policy=policy,
        )
        for name, policy in policies.items()
    )


DEFAULT_TYPE_REGISTRY = TypeRegistry(_default_types())
