from __future__ import annotations

from pathlib import Path
from uuid import UUID

from hypothesis import given, settings, strategies as st
import pytest

from comfyng.graph.cache import node_cache_key
from comfyng.graph.compiler import CompileContext, GraphCompilationError, GraphCompiler
from comfyng.graph.types import Edge, Graph, NodeInstance, OutputBinding
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import NodeDefinition


MAX_NODES = 7
PORTS = tuple(f"in_{index}" for index in range(MAX_NODES))
PROPERTY_NODE = NodeDefinition(
    id="ng.test.property",
    version="1.0.0",
    display_name="Property Node",
    package_id="org.comfyng.tests",
    input_schema_path=Path("/tmp/property.input.json"),
    output_schema_path=Path("/tmp/property.output.json"),
    input_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {name: {"type": "integer"} for name in PORTS},
        "required": [],
        "x-comfyng-optional": list(PORTS),
    },
    output_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    },
)
PROPERTY_CONTEXT = CompileContext(
    catalogue=NodeCatalogue(manifests=(), nodes=(PROPERTY_NODE,))
)


@st.composite
def _dag_cases(
    draw: st.DrawFn,
) -> tuple[int, tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    count = draw(st.integers(min_value=1, max_value=MAX_NODES))
    candidates = tuple(
        (source, target)
        for source in range(count)
        for target in range(source + 1, count)
    )
    if candidates:
        selected = tuple(
            sorted(
                draw(
                    st.sets(
                        st.sampled_from(candidates),
                        max_size=min(len(candidates), 12),
                    )
                )
            )
        )
    else:
        selected = ()
    node_order = draw(st.permutations(tuple(range(count))))
    edge_order = draw(st.permutations(tuple(range(len(selected)))))
    return count, selected, node_order, edge_order


def _graph(
    count: int,
    edges: tuple[tuple[int, int], ...],
    *,
    node_order: tuple[int, ...] | None = None,
    edge_order: tuple[int, ...] | None = None,
) -> Graph:
    nodes = tuple(
        NodeInstance(
            id=UUID(int=index + 1),
            type_id=PROPERTY_NODE.id,
            type_version=PROPERTY_NODE.version,
        )
        for index in range(count)
    )
    built_edges = tuple(
        Edge(
            source_node_id=nodes[source].id,
            source_port="value",
            target_node_id=nodes[target].id,
            target_port=f"in_{source}",
        )
        for source, target in edges
    )
    node_order = node_order or tuple(range(count))
    edge_order = edge_order or tuple(range(len(edges)))
    return Graph(
        id=UUID(int=10_000),
        version=1,
        nodes=tuple(nodes[index] for index in node_order),
        edges=tuple(built_edges[index] for index in edge_order),
        outputs={"result": OutputBinding(nodes[-1].id, "value")},
    )


@given(_dag_cases())
@settings(max_examples=60, deadline=None)
def test_generated_dags_compile_deterministically_across_storage_permutations(
    case: tuple[int, tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]],
) -> None:
    count, edges, node_order, edge_order = case
    canonical = GraphCompiler.compile(_graph(count, edges), PROPERTY_CONTEXT)
    permuted = GraphCompiler.compile(
        _graph(
            count,
            edges,
            node_order=node_order,
            edge_order=edge_order,
        ),
        PROPERTY_CONTEXT,
    )

    assert permuted.topological_order == canonical.topological_order
    assert permuted.cache_key == canonical.cache_key
    assert {step.node_id: step.cache_key for step in permuted.steps} == {
        step.node_id: step.cache_key for step in canonical.steps
    }


@given(st.integers(min_value=2, max_value=MAX_NODES))
@settings(max_examples=20, deadline=None)
def test_generated_directed_cycles_are_always_rejected(count: int) -> None:
    edges = tuple((index, index + 1) for index in range(count - 1)) + ((count - 1, 0),)

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(_graph(count, edges), PROPERTY_CONTEXT)

    assert "cycle" in {item.code for item in captured.value.diagnostics}


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.integers(min_value=-(2**63), max_value=2**63 - 1),
        max_size=8,
    )
)
@settings(max_examples=40, deadline=None)
def test_cache_keys_are_stable_for_equivalent_mapping_orders(
    values: dict[str, int],
) -> None:
    first = NodeInstance(
        id=UUID(int=1),
        type_id=PROPERTY_NODE.id,
        type_version=PROPERTY_NODE.version,
        inputs={"in_0": values},
    )
    second = NodeInstance(
        id=UUID(int=2),
        type_id=PROPERTY_NODE.id,
        type_version=PROPERTY_NODE.version,
        inputs={"in_0": dict(reversed(tuple(values.items())))},
    )

    assert node_cache_key(first) == node_cache_key(second)
