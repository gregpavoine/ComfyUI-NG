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


@st.composite
def _invalid_graph_cases(draw: st.DrawFn) -> tuple[Graph, str]:
    family = draw(
        st.sampled_from(
            (
                "duplicate",
                "missing_source",
                "missing_target",
                "missing_source_port",
                "missing_target_port",
                "unknown_version",
                "conflicting_binding",
            )
        )
    )
    first_id = UUID(int=draw(st.integers(min_value=1, max_value=2**32)))
    second_id = UUID(int=draw(st.integers(min_value=2**32 + 1, max_value=2**48)))
    missing_id = UUID(int=draw(st.integers(min_value=2**48 + 1, max_value=2**64)))
    first = NodeInstance(
        id=first_id,
        type_id=PROPERTY_NODE.id,
        type_version=PROPERTY_NODE.version,
    )
    second = NodeInstance(
        id=second_id,
        type_id=PROPERTY_NODE.id,
        type_version=PROPERTY_NODE.version,
    )
    nodes = (first, second)
    edges: tuple[Edge, ...] = ()
    expected = ""
    if family == "duplicate":
        nodes = (
            first,
            NodeInstance(
                id=first.id,
                type_id=PROPERTY_NODE.id,
                type_version=PROPERTY_NODE.version,
            ),
        )
        expected = "duplicate_node_id"
    elif family == "missing_source":
        edges = (Edge(missing_id, "value", second.id, "in_0"),)
        expected = "missing_source_node"
    elif family == "missing_target":
        edges = (Edge(first.id, "value", missing_id, "in_0"),)
        expected = "missing_target_node"
    elif family == "missing_source_port":
        edges = (Edge(first.id, "unknown", second.id, "in_0"),)
        expected = "missing_source_port"
    elif family == "missing_target_port":
        edges = (Edge(first.id, "value", second.id, "unknown"),)
        expected = "missing_target_port"
    elif family == "unknown_version":
        nodes = (
            NodeInstance(
                id=first.id,
                type_id=PROPERTY_NODE.id,
                type_version="9.9.9",
            ),
        )
        expected = "unknown_node_definition"
    else:
        second = NodeInstance(
            id=second.id,
            type_id=PROPERTY_NODE.id,
            type_version=PROPERTY_NODE.version,
            inputs={"in_0": 5},
        )
        nodes = (first, second)
        edges = (Edge(first.id, "value", second.id, "in_0"),)
        expected = "multiple_input_bindings"
    return Graph(id=UUID(int=90_000), version=1, nodes=nodes, edges=edges), expected


@given(_invalid_graph_cases())
@settings(max_examples=70, deadline=None)
def test_generated_structural_and_binding_failures_are_structured(
    case: tuple[Graph, str],
) -> None:
    graph, expected_code = case

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(graph, PROPERTY_CONTEXT)

    assert expected_code in {item.code for item in captured.value.diagnostics}


@given(st.integers(min_value=2, max_value=24))
@settings(max_examples=20, deadline=None)
def test_generated_excessive_loop_bounds_are_rejected(item_count: int) -> None:
    loop = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.for_each",
        type_version="1.0.0",
        inputs={"items": tuple(range(item_count))},
    )
    context = CompileContext(
        catalogue=NodeCatalogue.discover(),
        max_loop_iterations=item_count - 1,
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            Graph(id=UUID(int=91_000), version=1, nodes=(loop,), edges=()),
            context,
        )

    assert "loop_bound_exceeded" in {item.code for item in captured.value.diagnostics}


@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=7, deadline=None)
def test_generated_subgraph_depth_overflow_is_structured(depth: int) -> None:
    registry: dict[tuple[str, str], Graph] = {}
    keys = tuple((f"ng.subgraph.level_{index}", "1.0.0") for index in range(depth))
    leaf = NodeInstance(
        id=UUID(int=10_000 + depth),
        type_id=PROPERTY_NODE.id,
        type_version=PROPERTY_NODE.version,
    )
    registry[keys[-1]] = Graph(
        id=UUID(int=20_000 + depth),
        version=1,
        nodes=(leaf,),
        edges=(),
    )
    for index in reversed(range(depth - 1)):
        child_call = NodeInstance(
            id=UUID(int=30_000 + index),
            type_id=keys[index + 1][0],
            type_version="1.0.0",
        )
        registry[keys[index]] = Graph(
            id=UUID(int=40_000 + index),
            version=1,
            nodes=(child_call,),
            edges=(),
        )
    root_call = NodeInstance(
        id=UUID(int=50_000),
        type_id=keys[0][0],
        type_version="1.0.0",
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            Graph(id=UUID(int=92_000), version=1, nodes=(root_call,), edges=()),
            CompileContext(
                catalogue=PROPERTY_CONTEXT.catalogue,
                subgraphs=registry,
                max_subgraph_depth=depth - 1,
            ),
        )

    assert "subgraph_depth_exceeded" in {
        item.code for item in captured.value.diagnostics
    }


@given(st.integers(min_value=1, max_value=2**32))
@settings(max_examples=12, deadline=None)
def test_generated_recursive_subgraphs_are_rejected(call_value: int) -> None:
    key = ("ng.subgraph.recursive", "1.0.0")
    nested = NodeInstance(
        id=UUID(int=call_value),
        type_id=key[0],
        type_version=key[1],
    )
    child = Graph(id=UUID(int=93_000), version=1, nodes=(nested,), edges=())
    root = NodeInstance(
        id=UUID(int=2**64 + call_value),
        type_id=key[0],
        type_version=key[1],
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            Graph(id=UUID(int=94_000), version=1, nodes=(root,), edges=()),
            CompileContext(
                catalogue=PROPERTY_CONTEXT.catalogue,
                subgraphs={key: child},
            ),
        )

    assert "recursive_subgraph" in {item.code for item in captured.value.diagnostics}


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
