from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from comfyng.graph.cache import node_cache_key
from comfyng.graph.compiler import CompileContext, GraphCompilationError, GraphCompiler
from comfyng.graph.types import (
    Edge,
    Graph,
    InputBinding,
    NodeInstance,
    OutputBinding,
)
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import NodeDefinition


INTEGER = {"type": "integer"}


def _definition(
    node_id: str,
    *,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...] = ("value",),
) -> NodeDefinition:
    stem = node_id.replace(".", "_")
    return NodeDefinition(
        id=node_id,
        version="1.0.0",
        display_name=node_id,
        package_id="org.comfyng.tests",
        input_schema_path=Path(f"/tmp/{stem}.input.json"),
        output_schema_path=Path(f"/tmp/{stem}.output.json"),
        input_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {name: INTEGER for name in inputs},
            "required": list(inputs),
        },
        output_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {name: INTEGER for name in outputs},
            "required": list(outputs),
        },
    )


SOURCE = _definition("ng.test.source", inputs=("value",))
UNARY = _definition("ng.test.unary", inputs=("value",))
BINARY = _definition("ng.test.binary", inputs=("left", "right"))
IDENTITY = _definition("ng.test.identity", inputs=("value",))
TEST_CATALOGUE = NodeCatalogue(
    manifests=(),
    nodes=(SOURCE, UNARY, BINARY, IDENTITY),
)
TEST_CONTEXT = CompileContext(catalogue=TEST_CATALOGUE)


def _node(
    value: int,
    definition: NodeDefinition,
    *,
    inputs: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=UUID(int=value),
        type_id=definition.id,
        type_version=definition.version,
        inputs=inputs or {},
        metadata=metadata or {},
    )


def _diamond_graph(*, reverse_storage: bool = False) -> Graph:
    root = _node(1, SOURCE, inputs={"value": 3})
    left = _node(2, UNARY)
    right = _node(3, UNARY)
    join = _node(4, BINARY)
    nodes = (root, left, right, join)
    edges = (
        Edge(root.id, "value", left.id, "value"),
        Edge(root.id, "value", right.id, "value"),
        Edge(left.id, "value", join.id, "left"),
        Edge(right.id, "value", join.id, "right"),
    )
    if reverse_storage:
        nodes = tuple(reversed(nodes))
        edges = tuple(reversed(edges))
    return Graph(
        id=UUID(int=99),
        version=1,
        nodes=nodes,
        edges=edges,
        outputs={"result": OutputBinding(join.id, "value")},
    )


def test_compiler_builds_deterministic_kahn_groups_for_parallel_fan_out_and_in() -> (
    None
):
    first = GraphCompiler.compile(_diamond_graph(), TEST_CONTEXT)
    second = GraphCompiler.compile(_diamond_graph(reverse_storage=True), TEST_CONTEXT)

    expected = tuple(UUID(int=value) for value in (1, 2, 3, 4))
    assert first.topological_order == expected
    assert second.topological_order == expected
    assert tuple(
        tuple(step.node_id for step in group.steps) for group in first.groups
    ) == ((expected[0],), (expected[1], expected[2]), (expected[3],))
    assert first.cache_key == second.cache_key


def test_compiler_marks_transitive_constants_and_critical_path() -> None:
    plan = GraphCompiler.compile(_diamond_graph(), TEST_CONTEXT)

    assert all(step.is_constant for step in plan.steps)
    assert plan.critical_path == (
        UUID(int=1),
        UUID(int=2),
        UUID(int=4),
    )
    lengths = {step.node_id: step.critical_path_length for step in plan.steps}
    assert lengths == {
        UUID(int=1): 3,
        UUID(int=2): 2,
        UUID(int=3): 2,
        UUID(int=4): 1,
    }


def test_graph_inputs_make_their_transitive_steps_dynamic() -> None:
    source = _node(1, IDENTITY)
    sink = _node(2, UNARY)
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(source, sink),
        edges=(Edge(source.id, "value", sink.id, "value"),),
        inputs={"value": InputBinding(source.id, "value")},
        outputs={"result": OutputBinding(sink.id, "value")},
    )

    plan = GraphCompiler.compile(graph, TEST_CONTEXT)

    assert [step.is_constant for step in plan.steps] == [False, False]


def test_node_cache_key_is_content_derived_and_canonical() -> None:
    first = _node(1, SOURCE, inputs={"value": 3}, metadata={"mode": "fast"})
    same_content = _node(999, SOURCE, inputs={"value": 3}, metadata={"mode": "fast"})
    changed = _node(1, SOURCE, inputs={"value": 4}, metadata={"mode": "fast"})

    first_key = node_cache_key(
        first,
        {"right": "b" * 64, "left": "a" * 64},
    )

    assert first_key == node_cache_key(
        same_content,
        {"left": "a" * 64, "right": "b" * 64},
    )
    assert first_key != node_cache_key(changed, {"left": "a" * 64, "right": "b" * 64})
    assert len(first_key) == 64


def test_compiler_annotates_manifest_resources() -> None:
    catalogue = NodeCatalogue.discover()
    node = NodeInstance(
        id=UUID(int=1),
        type_id="ng.latent.empty",
        type_version="1.0.0",
        inputs={"width": 1024, "height": 1024},
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(node,),
        edges=(),
        outputs={"latent": OutputBinding(node.id, "latent")},
    )

    plan = GraphCompiler.compile(graph, CompileContext(catalogue=catalogue))

    assert plan.steps[0].resources.gpu == "optional"
    assert plan.steps[0].resources.estimated_ram_mb == 128
    assert plan.steps[0].resources.estimated_vram_mb == 256
    assert plan.peak_resources == plan.groups[0].resources


def test_compiler_expands_subgraph_calls_and_rewires_ports() -> None:
    inner = _node(10, IDENTITY)
    subgraph = Graph(
        id=UUID(int=500),
        version=1,
        nodes=(inner,),
        edges=(),
        inputs={"input": InputBinding(inner.id, "value")},
        outputs={"output": OutputBinding(inner.id, "value")},
    )
    source = _node(1, SOURCE, inputs={"value": 7})
    call = NodeInstance(
        id=UUID(int=2),
        type_id="ng.subgraph.identity",
        type_version="1.0.0",
    )
    sink = _node(3, UNARY)
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(source, call, sink),
        edges=(
            Edge(source.id, "value", call.id, "input"),
            Edge(call.id, "output", sink.id, "value"),
        ),
        outputs={"result": OutputBinding(sink.id, "value")},
    )
    context = CompileContext(
        catalogue=TEST_CATALOGUE,
        subgraphs={(call.type_id, call.type_version): subgraph},
    )

    plan = GraphCompiler.compile(graph, context)

    assert len(plan.steps) == 3
    assert call.id not in plan.topological_order
    assert [step.node_type_id for step in plan.steps] == [
        SOURCE.id,
        IDENTITY.id,
        UNARY.id,
    ]


def test_compiler_rejects_invalid_internal_subgraph_edges_with_a_diagnostic() -> None:
    inner = _node(10, IDENTITY)
    subgraph = Graph(
        id=UUID(int=500),
        version=1,
        nodes=(inner,),
        edges=(Edge(UUID(int=999), "value", inner.id, "value"),),
        outputs={"output": OutputBinding(inner.id, "value")},
    )
    call = NodeInstance(
        id=UUID(int=2),
        type_id="ng.subgraph.invalid",
        type_version="1.0.0",
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(call,),
        edges=(),
        outputs={"result": OutputBinding(call.id, "output")},
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            graph,
            CompileContext(
                catalogue=TEST_CATALOGUE,
                subgraphs={(call.type_id, call.type_version): subgraph},
            ),
        )

    assert "invalid_subgraph_edge" in {item.code for item in captured.value.diagnostics}


def test_compiler_annotates_bounded_loops_and_literal_conditions() -> None:
    catalogue = NodeCatalogue.discover()
    loop = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.for_each",
        type_version="1.0.0",
        inputs={"items": [1, 2, 3]},
    )
    condition = NodeInstance(
        id=UUID(int=2),
        type_id="ng.control.switch",
        type_version="1.0.0",
        inputs={"condition": True, "true_value": 1, "false_value": 0},
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(condition, loop),
        edges=(),
        outputs={
            "item": OutputBinding(loop.id, "item"),
            "value": OutputBinding(condition.id, "value"),
        },
    )

    plan = GraphCompiler.compile(
        graph,
        CompileContext(catalogue=catalogue, max_loop_iterations=3),
    )
    controls = {step.node_id: step.control for step in plan.steps}

    assert controls[loop.id] is not None
    assert controls[loop.id].kind == "loop"
    assert controls[loop.id].max_iterations == 3
    assert controls[condition.id] is not None
    assert controls[condition.id].kind == "condition"
    assert controls[condition.id].selected_branch == "true_value"


def test_compiler_rejects_a_loop_above_the_compile_bound() -> None:
    catalogue = NodeCatalogue.discover()
    loop = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.for_each",
        type_version="1.0.0",
        inputs={"items": [1, 2, 3]},
    )
    graph = Graph(id=uuid4(), version=1, nodes=(loop,), edges=())

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            graph,
            CompileContext(catalogue=catalogue, max_loop_iterations=2),
        )

    assert {
        item.code for item in captured.value.diagnostics if item.severity == "error"
    } == {"loop_bound_exceeded"}


def test_compiler_rejects_cycles() -> None:
    left = _node(1, UNARY)
    right = _node(2, UNARY)
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(left, right),
        edges=(
            Edge(left.id, "value", right.id, "value"),
            Edge(right.id, "value", left.id, "value"),
        ),
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(graph, TEST_CONTEXT)

    assert "cycle" in {item.code for item in captured.value.diagnostics}
