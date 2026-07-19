from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from uuid import UUID, uuid5

from comfyng.core.contracts import Contract, register_contract
from comfyng.core.ids import validate_node_id, validate_semver

from .types import Edge, Graph, InputBinding, NodeInstance, OutputBinding


SubgraphKey = tuple[str, str]
SUBGRAPH_CONSTRUCT_KINDS = frozenset(("subgraph", "function", "macro"))


@register_contract
class SubgraphTrace(Contract):
    """Trace from a logical graph construct to its expanded execution nodes."""

    TYPE_ID: ClassVar[str] = "comfyng.subgraph-trace"

    kind: str
    call_node_id: UUID
    parent_call_node_id: UUID | None
    node_type_id: str
    node_type_version: str
    source_graph_id: UUID
    source_graph_version: int
    member_node_ids: tuple[UUID, ...]
    depth: int

    def __post_init__(self) -> None:
        if self.kind not in SUBGRAPH_CONSTRUCT_KINDS:
            raise ValueError("subgraph trace kind must be subgraph, function or macro")
        for field_name in ("call_node_id", "source_graph_id"):
            if not isinstance(getattr(self, field_name), UUID):
                raise ValueError(f"{field_name} must be a UUID")
        if self.parent_call_node_id is not None and not isinstance(
            self.parent_call_node_id, UUID
        ):
            raise ValueError("parent_call_node_id must be a UUID")
        if self.parent_call_node_id == self.call_node_id:
            raise ValueError("a subgraph trace cannot be its own parent")
        validate_node_id(self.node_type_id)
        validate_semver(self.node_type_version, field="node_type_version")
        if type(self.source_graph_version) is not int or self.source_graph_version <= 0:
            raise ValueError("source_graph_version must be positive")
        if not self.member_node_ids:
            raise ValueError("subgraph trace must contain expanded member nodes")
        if any(not isinstance(node_id, UUID) for node_id in self.member_node_ids):
            raise ValueError("member_node_ids must contain UUID values")
        if len(set(self.member_node_ids)) != len(self.member_node_ids):
            raise ValueError("member_node_ids must be unique")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("subgraph trace depth must be non-negative")
        if (self.parent_call_node_id is None) != (self.depth == 0):
            raise ValueError("only root subgraph traces may have depth zero")


@dataclass(frozen=True, slots=True)
class SubgraphExpansion:
    graph: Graph
    traces: tuple[SubgraphTrace, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Graph):
            raise TypeError("graph must be a Graph")
        if any(not isinstance(trace, SubgraphTrace) for trace in self.traces):
            raise TypeError("traces must contain SubgraphTrace values")


class SubgraphExpansionError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        node_id: UUID | None = None,
    ) -> None:
        self.code = code
        self.node_id = node_id
        super().__init__(message)


def _registry(
    subgraphs: Mapping[SubgraphKey | str, Graph],
) -> dict[SubgraphKey, Graph]:
    resolved: dict[SubgraphKey, Graph] = {}
    for raw_key, graph in subgraphs.items():
        if isinstance(raw_key, str):
            key = (raw_key, "1.0.0")
        elif (
            isinstance(raw_key, tuple)
            and len(raw_key) == 2
            and all(isinstance(item, str) for item in raw_key)
        ):
            key = raw_key
        else:
            raise TypeError(
                "subgraph keys must be type ids or (type id, version) pairs"
            )
        if not isinstance(graph, Graph):
            raise TypeError("subgraph registry values must be Graph values")
        if key in resolved:
            raise ValueError(f"duplicate subgraph registration {key[0]}@{key[1]}")
        resolved[key] = graph
    return resolved


def _cloned_id(call_id: UUID, graph: Graph, inner_id: UUID) -> UUID:
    return uuid5(call_id, f"{graph.id}:{graph.version}:{inner_id}")


def _clone_binding_input(
    binding: InputBinding,
    node_ids: Mapping[UUID, UUID],
    *,
    call_id: UUID,
) -> InputBinding:
    try:
        return InputBinding(node_ids[binding.node_id], binding.port)
    except KeyError as exc:
        raise SubgraphExpansionError(
            "invalid_subgraph_binding",
            "subgraph input references a missing node",
            node_id=call_id,
        ) from exc


def _clone_binding_output(
    binding: OutputBinding,
    node_ids: Mapping[UUID, UUID],
    *,
    call_id: UUID,
) -> OutputBinding:
    try:
        return OutputBinding(node_ids[binding.node_id], binding.port)
    except KeyError as exc:
        raise SubgraphExpansionError(
            "invalid_subgraph_binding",
            "subgraph output references a missing node",
            node_id=call_id,
        ) from exc


def _inline_call(graph: Graph, call: NodeInstance, child: Graph) -> Graph:
    cloned_ids = {node.id: _cloned_id(call.id, child, node.id) for node in child.nodes}
    existing_ids = {node.id for node in graph.nodes if node.id != call.id}
    collisions = existing_ids.intersection(cloned_ids.values())
    if collisions:
        raise SubgraphExpansionError(
            "subgraph_node_collision",
            "expanded subgraph node ids collide with the parent graph",
            node_id=call.id,
        )

    literal_overrides: dict[UUID, dict[str, object]] = {}
    for port, value in call.inputs.items():
        binding = child.inputs.get(port)
        if binding is None:
            raise SubgraphExpansionError(
                "unknown_subgraph_input",
                f"subgraph call has no input port {port!r}",
                node_id=call.id,
            )
        overrides = literal_overrides.setdefault(binding.node_id, {})
        overrides[binding.port] = value

    cloned_nodes: list[NodeInstance] = []
    for node in child.nodes:
        inputs = dict(node.inputs)
        for port, value in literal_overrides.get(node.id, {}).items():
            if port in inputs:
                raise SubgraphExpansionError(
                    "multiple_subgraph_input",
                    f"subgraph input {port!r} is already bound internally",
                    node_id=call.id,
                )
            inputs[port] = value
        cloned_nodes.append(
            NodeInstance(
                id=cloned_ids[node.id],
                type_id=node.type_id,
                type_version=node.type_version,
                inputs=inputs,
                metadata=node.metadata,
            )
        )

    incoming_ports = {
        edge.target_port for edge in graph.edges if edge.target_node_id == call.id
    }
    duplicate_ports = incoming_ports.intersection(call.inputs)
    if duplicate_ports:
        port = min(duplicate_ports)
        raise SubgraphExpansionError(
            "multiple_subgraph_input",
            f"subgraph input {port!r} has both a literal and an edge",
            node_id=call.id,
        )

    rewritten_edges: list[Edge] = []
    for edge in graph.edges:
        if edge.target_node_id == call.id:
            binding = child.inputs.get(edge.target_port)
            if binding is None:
                raise SubgraphExpansionError(
                    "unknown_subgraph_input",
                    f"subgraph call has no input port {edge.target_port!r}",
                    node_id=call.id,
                )
            cloned = _clone_binding_input(binding, cloned_ids, call_id=call.id)
            rewritten_edges.append(
                Edge(
                    edge.source_node_id,
                    edge.source_port,
                    cloned.node_id,
                    cloned.port,
                )
            )
        elif edge.source_node_id == call.id:
            binding = child.outputs.get(edge.source_port)
            if binding is None:
                raise SubgraphExpansionError(
                    "unknown_subgraph_output",
                    f"subgraph call has no output port {edge.source_port!r}",
                    node_id=call.id,
                )
            cloned = _clone_binding_output(binding, cloned_ids, call_id=call.id)
            rewritten_edges.append(
                Edge(
                    cloned.node_id,
                    cloned.port,
                    edge.target_node_id,
                    edge.target_port,
                )
            )
        else:
            rewritten_edges.append(edge)
    for edge in child.edges:
        try:
            source_id = cloned_ids[edge.source_node_id]
            target_id = cloned_ids[edge.target_node_id]
        except KeyError as exc:
            raise SubgraphExpansionError(
                "invalid_subgraph_edge",
                "subgraph edge references a missing node",
                node_id=call.id,
            ) from exc
        rewritten_edges.append(
            Edge(source_id, edge.source_port, target_id, edge.target_port)
        )

    rewritten_inputs: dict[str, InputBinding] = {}
    for name, binding in graph.inputs.items():
        if binding.node_id != call.id:
            rewritten_inputs[name] = binding
            continue
        child_binding = child.inputs.get(binding.port)
        if child_binding is None:
            raise SubgraphExpansionError(
                "unknown_subgraph_input",
                f"subgraph call has no input port {binding.port!r}",
                node_id=call.id,
            )
        rewritten_inputs[name] = _clone_binding_input(
            child_binding,
            cloned_ids,
            call_id=call.id,
        )

    rewritten_outputs: dict[str, OutputBinding] = {}
    for name, binding in graph.outputs.items():
        if binding.node_id != call.id:
            rewritten_outputs[name] = binding
            continue
        child_binding = child.outputs.get(binding.port)
        if child_binding is None:
            raise SubgraphExpansionError(
                "unknown_subgraph_output",
                f"subgraph call has no output port {binding.port!r}",
                node_id=call.id,
            )
        rewritten_outputs[name] = _clone_binding_output(
            child_binding,
            cloned_ids,
            call_id=call.id,
        )

    parent_nodes = tuple(node for node in graph.nodes if node.id != call.id)
    return Graph(
        id=graph.id,
        version=graph.version,
        nodes=parent_nodes + tuple(cloned_nodes),
        edges=tuple(rewritten_edges),
        inputs=rewritten_inputs,
        outputs=rewritten_outputs,
    )


def _trace_kind(call: NodeInstance) -> str:
    value = call.metadata.get("construct_kind", "subgraph")
    if not isinstance(value, str) or value not in SUBGRAPH_CONSTRUCT_KINDS:
        raise SubgraphExpansionError(
            "invalid_subgraph_kind",
            "subgraph construct_kind must be subgraph, function or macro",
            node_id=call.id,
        )
    return value


def _rebased_trace(
    trace: SubgraphTrace,
    *,
    outer_call: NodeInstance,
    expanded_child: Graph,
) -> SubgraphTrace:
    def namespace(value: UUID) -> UUID:
        return _cloned_id(outer_call.id, expanded_child, value)

    return SubgraphTrace(
        kind=trace.kind,
        call_node_id=namespace(trace.call_node_id),
        parent_call_node_id=(
            outer_call.id
            if trace.parent_call_node_id is None
            else namespace(trace.parent_call_node_id)
        ),
        node_type_id=trace.node_type_id,
        node_type_version=trace.node_type_version,
        source_graph_id=trace.source_graph_id,
        source_graph_version=trace.source_graph_version,
        member_node_ids=tuple(namespace(node_id) for node_id in trace.member_node_ids),
        depth=trace.depth + 1,
    )


def _expand(
    graph: Graph,
    registry: Mapping[SubgraphKey, Graph],
    *,
    stack: tuple[SubgraphKey, ...],
    max_depth: int,
) -> SubgraphExpansion:
    if len(stack) > max_depth:
        raise SubgraphExpansionError(
            "subgraph_depth_exceeded",
            f"subgraph expansion exceeds maximum depth {max_depth}",
        )
    expanded = graph
    traces: list[SubgraphTrace] = []
    while True:
        calls = tuple(
            node
            for node in expanded.nodes
            if (node.type_id, node.type_version) in registry
        )
        call = min(calls, key=lambda node: str(node.id)) if calls else None
        if call is None:
            return SubgraphExpansion(expanded, tuple(traces))
        key = (call.type_id, call.type_version)
        if key in stack:
            raise SubgraphExpansionError(
                "recursive_subgraph",
                f"recursive subgraph call {key[0]}@{key[1]}",
                node_id=call.id,
            )
        child_expansion = _expand(
            registry[key],
            registry,
            stack=stack + (key,),
            max_depth=max_depth,
        )
        child = child_expansion.graph
        cloned_member_ids = tuple(
            _cloned_id(call.id, child, node.id) for node in child.nodes
        )
        if not cloned_member_ids:
            raise SubgraphExpansionError(
                "empty_subgraph",
                "subgraph constructs must expand to at least one node",
                node_id=call.id,
            )
        trace = SubgraphTrace(
            kind=_trace_kind(call),
            call_node_id=call.id,
            parent_call_node_id=None,
            node_type_id=call.type_id,
            node_type_version=call.type_version,
            source_graph_id=child.id,
            source_graph_version=child.version,
            member_node_ids=cloned_member_ids,
            depth=0,
        )
        traces.append(trace)
        traces.extend(
            _rebased_trace(
                child_trace,
                outer_call=call,
                expanded_child=child,
            )
            for child_trace in child_expansion.traces
        )
        expanded = _inline_call(expanded, call, child)


def expand_subgraphs(
    graph: Graph,
    subgraphs: Mapping[SubgraphKey | str, Graph],
    *,
    max_depth: int = 32,
) -> Graph:
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    if type(max_depth) is not int or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")
    return _expand(
        graph,
        _registry(subgraphs),
        stack=(),
        max_depth=max_depth,
    ).graph


def expand_subgraphs_with_trace(
    graph: Graph,
    subgraphs: Mapping[SubgraphKey | str, Graph],
    *,
    max_depth: int = 32,
) -> SubgraphExpansion:
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    if type(max_depth) is not int or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")
    return _expand(graph, _registry(subgraphs), stack=(), max_depth=max_depth)
