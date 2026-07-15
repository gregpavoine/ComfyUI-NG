from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from uuid import UUID

from .types import Edge, Graph, NodeInstance


class CycleError(ValueError):
    """Raised when a deterministic topological traversal finds a cycle."""

    def __init__(self, node_ids: Iterable[UUID]) -> None:
        self.node_ids = tuple(sorted(node_ids, key=str))
        rendered = ", ".join(str(node_id) for node_id in self.node_ids)
        super().__init__(f"graph contains a cycle involving: {rendered}")


def _parts(
    graph_or_nodes: Graph | Iterable[NodeInstance | UUID],
    edges: Iterable[Edge] | None,
) -> tuple[tuple[UUID, ...], tuple[Edge, ...]]:
    if isinstance(graph_or_nodes, Graph):
        if edges is not None:
            raise TypeError("edges must be omitted when a Graph is provided")
        return (
            tuple(node.id for node in graph_or_nodes.nodes),
            graph_or_nodes.edges,
        )
    resolved_nodes = tuple(
        value.id if isinstance(value, NodeInstance) else value
        for value in graph_or_nodes
    )
    if any(not isinstance(value, UUID) for value in resolved_nodes):
        raise TypeError("nodes must contain UUID or NodeInstance values")
    return resolved_nodes, tuple(edges or ())


def topological_layers(
    graph_or_nodes: Graph | Iterable[NodeInstance | UUID],
    edges: Iterable[Edge] | None = None,
) -> tuple[tuple[UUID, ...], ...]:
    """Return deterministic Kahn layers, exposing independent parallel work."""

    node_ids, resolved_edges = _parts(graph_or_nodes, edges)
    node_set = set(node_ids)
    if len(node_set) != len(node_ids):
        raise ValueError("topological sorting requires unique node ids")

    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[UUID, list[UUID]] = defaultdict(list)
    for edge in resolved_edges:
        if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
            raise ValueError("topological sorting requires valid edge endpoints")
        outgoing[edge.source_node_id].append(edge.target_node_id)
        indegree[edge.target_node_id] += 1

    for targets in outgoing.values():
        targets.sort(key=str)

    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=str,
    )
    layers: list[tuple[UUID, ...]] = []
    visited = 0
    while ready:
        layer = tuple(ready)
        layers.append(layer)
        visited += len(layer)
        next_ready: list[UUID] = []
        for node_id in layer:
            for target_id in outgoing.get(node_id, ()):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    next_ready.append(target_id)
        ready = sorted(next_ready, key=str)

    if visited != len(node_ids):
        raise CycleError(node_id for node_id, degree in indegree.items() if degree > 0)
    return tuple(layers)


def topological_sort(
    graph_or_nodes: Graph | Iterable[NodeInstance | UUID],
    edges: Iterable[Edge] | None = None,
) -> tuple[UUID, ...]:
    return tuple(
        node_id
        for layer in topological_layers(graph_or_nodes, edges)
        for node_id in layer
    )


def critical_path_lengths(
    order: Iterable[UUID],
    edges: Iterable[Edge],
) -> dict[UUID, int]:
    resolved_order = tuple(order)
    outgoing: dict[UUID, set[UUID]] = defaultdict(set)
    for edge in edges:
        outgoing[edge.source_node_id].add(edge.target_node_id)
    lengths: dict[UUID, int] = {}
    for node_id in reversed(resolved_order):
        targets = outgoing.get(node_id, set())
        lengths[node_id] = 1 + max(
            (lengths[target_id] for target_id in targets),
            default=0,
        )
    return lengths


def select_critical_path(
    order: Iterable[UUID],
    edges: Iterable[Edge],
    lengths: dict[UUID, int] | None = None,
) -> tuple[UUID, ...]:
    resolved_order = tuple(order)
    if not resolved_order:
        return ()
    resolved_lengths = lengths or critical_path_lengths(resolved_order, edges)
    outgoing: dict[UUID, set[UUID]] = defaultdict(set)
    indegree = {node_id: 0 for node_id in resolved_order}
    for edge in edges:
        outgoing[edge.source_node_id].add(edge.target_node_id)
        indegree[edge.target_node_id] += 1
    roots = tuple(node_id for node_id in resolved_order if indegree[node_id] == 0)
    current = min(
        roots,
        key=lambda node_id: (-resolved_lengths[node_id], str(node_id)),
    )
    path = [current]
    while resolved_lengths[current] > 1:
        candidates = tuple(
            target_id
            for target_id in outgoing.get(current, set())
            if resolved_lengths[target_id] == resolved_lengths[current] - 1
        )
        if not candidates:
            break
        current = min(candidates, key=str)
        path.append(current)
    return tuple(path)
