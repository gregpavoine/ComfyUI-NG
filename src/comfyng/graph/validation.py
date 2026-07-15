from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol
from uuid import UUID

from jsonschema import Draft202012Validator

from comfyng.core.contracts import Contract, register_contract
from comfyng.core.errors import UnknownNodeDefinitionError, UnknownTypeDefinitionError
from comfyng.core.ids import validate_port_name
from comfyng.core.json_values import validate_safe_unicode_string
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import NodeDefinition

from .topology import CycleError, topological_layers
from .types import Edge, Graph, TypeRef, TypeRegistry


class GraphValidationContext(Protocol):
    catalogue: NodeCatalogue
    type_registry: TypeRegistry
    max_loop_iterations: int


@register_contract
class GraphDiagnostic(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.graph-diagnostic"

    code: str
    severity: str
    message: str
    node_id: UUID | None = None
    edge_index: int | None = None
    port: str | None = None

    def __post_init__(self) -> None:
        for field in ("code", "message"):
            value = validate_safe_unicode_string(getattr(self, field), field=field)
            if not value:
                raise ValueError(f"{field} must be non-empty")
        if self.severity not in ("error", "warning"):
            raise ValueError("diagnostic severity must be error or warning")
        if self.edge_index is not None and (
            type(self.edge_index) is not int or self.edge_index < 0
        ):
            raise ValueError("edge_index must be non-negative")
        if self.port is not None:
            validate_port_name(self.port)


def _diagnostic(
    code: str,
    message: str,
    *,
    severity: str = "error",
    node_id: UUID | None = None,
    edge_index: int | None = None,
    port: str | None = None,
) -> GraphDiagnostic:
    return GraphDiagnostic(
        code=code,
        severity=severity,
        message=message,
        node_id=node_id,
        edge_index=edge_index,
        port=port,
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _type_ref(schema: Mapping[str, Any]) -> TypeRef | None:
    value = schema.get("x-comfyng-type")
    if value is None:
        return None
    return TypeRef.parse(value)


_ALL_JSON_TYPES = frozenset(
    ("null", "boolean", "object", "array", "number", "string", "integer")
)


def _json_types(schema: Mapping[str, Any]) -> frozenset[str] | None:
    value = schema.get("type")
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset((value,))
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, str) for item in value
    ):
        return frozenset(value)
    return frozenset()


def _is_wildcard(schema: Mapping[str, Any]) -> bool:
    types = _json_types(schema)
    return types is None or _ALL_JSON_TYPES.issubset(types)


def _json_compatible(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    source_types = _json_types(source)
    target_types = _json_types(target)
    if source_types is None or target_types is None:
        return True
    for source_type in source_types:
        if source_type == "integer" and (
            "integer" in target_types or "number" in target_types
        ):
            continue
        if source_type not in target_types:
            return False
    return True


def _edge_type_diagnostics(
    edge: Edge,
    edge_index: int,
    source_schema: Mapping[str, Any],
    target_schema: Mapping[str, Any],
    registry: TypeRegistry,
) -> list[GraphDiagnostic]:
    diagnostics: list[GraphDiagnostic] = []
    try:
        source_ref = _type_ref(source_schema)
        target_ref = _type_ref(target_schema)
    except ValueError as exc:
        return [
            _diagnostic(
                "invalid_type_reference",
                str(exc),
                edge_index=edge_index,
                port=edge.target_port,
            )
        ]

    unknown = False
    for ref in (source_ref, target_ref):
        if ref is None:
            continue
        try:
            registry.resolve_ref(ref)
        except UnknownTypeDefinitionError as exc:
            diagnostics.append(
                _diagnostic(
                    "unknown_type",
                    str(exc),
                    edge_index=edge_index,
                    port=edge.target_port,
                )
            )
            unknown = True
    if unknown:
        return diagnostics

    if source_ref is not None and target_ref is not None:
        if source_ref.name != target_ref.name:
            diagnostics.append(
                _diagnostic(
                    "incompatible_types",
                    f"{source_ref} cannot connect to {target_ref}",
                    edge_index=edge_index,
                    port=edge.target_port,
                )
            )
        elif source_ref.version != target_ref.version:
            diagnostics.append(
                _diagnostic(
                    "type_version_mismatch",
                    f"{source_ref} cannot connect to {target_ref}",
                    edge_index=edge_index,
                    port=edge.target_port,
                )
            )
        return diagnostics

    if source_ref is not None or target_ref is not None:
        generic = target_schema if source_ref is not None else source_schema
        if not _is_wildcard(generic):
            diagnostics.append(
                _diagnostic(
                    "incompatible_types",
                    "typed and untyped ports are not compatible",
                    edge_index=edge_index,
                    port=edge.target_port,
                )
            )
        return diagnostics

    if not _json_compatible(source_schema, target_schema):
        diagnostics.append(
            _diagnostic(
                "incompatible_types",
                "source JSON type is not accepted by target port",
                edge_index=edge_index,
                port=edge.target_port,
            )
        )
    return diagnostics


def _sort_key(diagnostic: GraphDiagnostic) -> tuple[object, ...]:
    return (
        0 if diagnostic.severity == "error" else 1,
        diagnostic.code,
        str(diagnostic.node_id) if diagnostic.node_id is not None else "",
        diagnostic.port or "",
        diagnostic.edge_index if diagnostic.edge_index is not None else -1,
        diagnostic.message,
    )


def validate_graph(
    graph: Graph,
    context: GraphValidationContext,
) -> tuple[GraphDiagnostic, ...]:
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    diagnostics: list[GraphDiagnostic] = []
    counts = Counter(node.id for node in graph.nodes)
    for node_id, count in sorted(counts.items(), key=lambda item: str(item[0])):
        if count > 1:
            diagnostics.append(
                _diagnostic(
                    "duplicate_node_id",
                    f"node id {node_id} occurs {count} times",
                    node_id=node_id,
                )
            )
    nodes = {node.id: node for node in graph.nodes}

    definitions: dict[UUID, NodeDefinition] = {}
    for node in sorted(graph.nodes, key=lambda item: str(item.id)):
        try:
            definitions[node.id] = context.catalogue.get(
                node.type_id,
                node.type_version,
            )
        except UnknownNodeDefinitionError:
            diagnostics.append(
                _diagnostic(
                    "unknown_node_definition",
                    f"unknown node definition {node.type_id}@{node.type_version}",
                    node_id=node.id,
                )
            )

    incoming: dict[tuple[UUID, str], list[tuple[int, Edge]]] = defaultdict(list)
    used_outputs: set[tuple[UUID, str]] = set()
    endpoint_edges: list[Edge] = []
    for edge_index, edge in sorted(
        enumerate(graph.edges),
        key=lambda item: (
            str(item[1].source_node_id),
            item[1].source_port,
            str(item[1].target_node_id),
            item[1].target_port,
            item[0],
        ),
    ):
        source = nodes.get(edge.source_node_id)
        target = nodes.get(edge.target_node_id)
        if source is None:
            diagnostics.append(
                _diagnostic(
                    "missing_source_node",
                    f"edge source node {edge.source_node_id} does not exist",
                    edge_index=edge_index,
                    port=edge.source_port,
                )
            )
        if target is None:
            diagnostics.append(
                _diagnostic(
                    "missing_target_node",
                    f"edge target node {edge.target_node_id} does not exist",
                    edge_index=edge_index,
                    port=edge.target_port,
                )
            )
        if source is None or target is None:
            continue
        endpoint_edges.append(edge)
        source_definition = definitions.get(source.id)
        target_definition = definitions.get(target.id)
        if source_definition is None or target_definition is None:
            continue
        source_properties = source_definition.output_schema.get("properties", {})
        target_properties = target_definition.input_schema.get("properties", {})
        source_schema = source_properties.get(edge.source_port)
        target_schema = target_properties.get(edge.target_port)
        if source_schema is None:
            diagnostics.append(
                _diagnostic(
                    "missing_source_port",
                    f"node {source.id} has no output port {edge.source_port!r}",
                    node_id=source.id,
                    edge_index=edge_index,
                    port=edge.source_port,
                )
            )
        else:
            used_outputs.add((source.id, edge.source_port))
        if target_schema is None:
            diagnostics.append(
                _diagnostic(
                    "missing_target_port",
                    f"node {target.id} has no input port {edge.target_port!r}",
                    node_id=target.id,
                    edge_index=edge_index,
                    port=edge.target_port,
                )
            )
        else:
            incoming[(target.id, edge.target_port)].append((edge_index, edge))
        if source_schema is not None and target_schema is not None:
            diagnostics.extend(
                _edge_type_diagnostics(
                    edge,
                    edge_index,
                    source_schema,
                    target_schema,
                    context.type_registry,
                )
            )

    external_inputs: dict[tuple[UUID, str], list[str]] = defaultdict(list)
    for name, binding in sorted(graph.inputs.items()):
        node = nodes.get(binding.node_id)
        if node is None:
            diagnostics.append(
                _diagnostic(
                    "missing_graph_input_node",
                    f"graph input {name!r} references a missing node",
                    node_id=binding.node_id,
                    port=binding.port,
                )
            )
            continue
        definition = definitions.get(node.id)
        if definition is not None and binding.port not in definition.input_schema.get(
            "properties", {}
        ):
            diagnostics.append(
                _diagnostic(
                    "missing_graph_input_port",
                    f"graph input {name!r} references unknown port {binding.port!r}",
                    node_id=node.id,
                    port=binding.port,
                )
            )
        else:
            external_inputs[(node.id, binding.port)].append(name)

    for name, binding in sorted(graph.outputs.items()):
        node = nodes.get(binding.node_id)
        if node is None:
            diagnostics.append(
                _diagnostic(
                    "missing_graph_output_node",
                    f"graph output {name!r} references a missing node",
                    node_id=binding.node_id,
                    port=binding.port,
                )
            )
            continue
        definition = definitions.get(node.id)
        if definition is not None and binding.port not in definition.output_schema.get(
            "properties", {}
        ):
            diagnostics.append(
                _diagnostic(
                    "missing_graph_output_port",
                    f"graph output {name!r} references unknown port {binding.port!r}",
                    node_id=node.id,
                    port=binding.port,
                )
            )
        else:
            used_outputs.add((node.id, binding.port))

    for node in sorted(graph.nodes, key=lambda item: str(item.id)):
        definition = definitions.get(node.id)
        if definition is None:
            continue
        properties = definition.input_schema.get("properties", {})
        for port, value in sorted(node.inputs.items()):
            schema = properties.get(port)
            if schema is None:
                diagnostics.append(
                    _diagnostic(
                        "unknown_literal_input",
                        f"node {node.id} has no input port {port!r}",
                        node_id=node.id,
                        port=port,
                    )
                )
                continue
            errors = tuple(Draft202012Validator(schema).iter_errors(_thaw(value)))
            if errors:
                diagnostics.append(
                    _diagnostic(
                        "invalid_literal_input",
                        f"invalid literal for {port!r}: {errors[0].message}",
                        node_id=node.id,
                        port=port,
                    )
                )
        required = tuple(definition.input_schema.get("required", ()))
        for port in sorted(required):
            sources = (
                int(port in node.inputs)
                + len(incoming.get((node.id, port), ()))
                + len(external_inputs.get((node.id, port), ()))
            )
            if sources == 0:
                diagnostics.append(
                    _diagnostic(
                        "missing_required_input",
                        f"required input {port!r} is not bound",
                        node_id=node.id,
                        port=port,
                    )
                )
            elif sources > 1:
                diagnostics.append(
                    _diagnostic(
                        "multiple_input_bindings",
                        f"input {port!r} has more than one binding",
                        node_id=node.id,
                        port=port,
                    )
                )

        for port in sorted(properties):
            sources = (
                int(port in node.inputs)
                + len(incoming.get((node.id, port), ()))
                + len(external_inputs.get((node.id, port), ()))
            )
            if sources > 1 and port not in required:
                diagnostics.append(
                    _diagnostic(
                        "multiple_input_bindings",
                        f"input {port!r} has more than one binding",
                        node_id=node.id,
                        port=port,
                    )
                )

        if node.type_id == "ng.control.for_each":
            if "items" in node.inputs and isinstance(node.inputs["items"], tuple):
                bound = len(node.inputs["items"])
            else:
                candidate = node.metadata.get("max_iterations")
                bound = candidate if type(candidate) is int and candidate > 0 else None
                if bound is None:
                    diagnostics.append(
                        _diagnostic(
                            "unbounded_loop",
                            "dynamic for-each nodes require metadata.max_iterations",
                            node_id=node.id,
                        )
                    )
            if bound is not None and bound > context.max_loop_iterations:
                diagnostics.append(
                    _diagnostic(
                        "loop_bound_exceeded",
                        f"loop bound {bound} exceeds {context.max_loop_iterations}",
                        node_id=node.id,
                    )
                )

        output_properties = definition.output_schema.get("properties", {})
        for port in sorted(output_properties):
            if (node.id, port) not in used_outputs:
                diagnostics.append(
                    _diagnostic(
                        "unused_output",
                        f"output {port!r} is not consumed",
                        severity="warning",
                        node_id=node.id,
                        port=port,
                    )
                )

    if len(nodes) == len(graph.nodes) and all(
        edge.source_node_id in nodes and edge.target_node_id in nodes
        for edge in endpoint_edges
    ):
        try:
            topological_layers(tuple(nodes), endpoint_edges)
        except CycleError as exc:
            diagnostics.append(
                _diagnostic(
                    "cycle",
                    str(exc),
                    node_id=exc.node_ids[0] if exc.node_ids else None,
                )
            )

    return tuple(sorted(diagnostics, key=_sort_key))
