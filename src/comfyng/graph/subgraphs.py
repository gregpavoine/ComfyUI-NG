from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID, uuid5

from .types import Edge, Graph, InputBinding, NodeInstance, OutputBinding


SubgraphKey = tuple[str, str]


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


def _expand(
    graph: Graph,
    registry: Mapping[SubgraphKey, Graph],
    *,
    stack: tuple[SubgraphKey, ...],
    max_depth: int,
) -> Graph:
    if len(stack) > max_depth:
        raise SubgraphExpansionError(
            "subgraph_depth_exceeded",
            f"subgraph expansion exceeds maximum depth {max_depth}",
        )
    expanded = graph
    while True:
        call = next(
            (
                node
                for node in expanded.nodes
                if (node.type_id, node.type_version) in registry
            ),
            None,
        )
        if call is None:
            return expanded
        key = (call.type_id, call.type_version)
        if key in stack:
            raise SubgraphExpansionError(
                "recursive_subgraph",
                f"recursive subgraph call {key[0]}@{key[1]}",
                node_id=call.id,
            )
        child = _expand(
            registry[key],
            registry,
            stack=stack + (key,),
            max_depth=max_depth,
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
    return _expand(graph, _registry(subgraphs), stack=(), max_depth=max_depth)
