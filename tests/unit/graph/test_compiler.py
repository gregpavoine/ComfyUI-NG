from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import msgspec
import pytest

from comfyng.core.enums import (
    GpuRequirement,
    LoadPolicy,
    RuntimeIsolation,
    UnloadPolicy,
)
from comfyng.graph.cache import node_cache_key
from comfyng.graph.compiler import (
    CompileContext,
    ControlBranch,
    ControlRegion,
    ExecutionPlan,
    GraphCompilationError,
    GraphCompiler,
)
from comfyng.graph.types import (
    Edge,
    Graph,
    InputBinding,
    NodeInstance,
    OutputBinding,
)
from comfyng.graph.subgraphs import SubgraphTrace
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import (
    NodeDefinition,
    NodeExecutionTraits,
    PackageMetadata,
    PluginManifest,
    ResourceRequirements,
    RuntimeDefinition,
)


INTEGER = {"type": "integer"}


def _definition(
    node_id: str,
    *,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...] = ("value",),
    execution: NodeExecutionTraits | None = None,
    package_id: str = "org.comfyng.tests",
    input_port_schema: dict[str, object] | None = None,
    output_port_schema: dict[str, object] | None = None,
) -> NodeDefinition:
    stem = node_id.replace(".", "_")
    return NodeDefinition(
        id=node_id,
        version="1.0.0",
        display_name=node_id,
        package_id=package_id,
        input_schema_path=Path(f"/tmp/{stem}.input.json"),
        output_schema_path=Path(f"/tmp/{stem}.output.json"),
        input_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {name: input_port_schema or INTEGER for name in inputs},
            "required": list(inputs),
        },
        output_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {name: output_port_schema or INTEGER for name in outputs},
            "required": list(outputs),
        },
        execution=execution or NodeExecutionTraits(),
    )


def _manifest(
    *definitions: NodeDefinition,
    package_id: str = "org.comfyng.tests",
    isolation: RuntimeIsolation = RuntimeIsolation.CPU_WORKER,
    gpu: GpuRequirement = GpuRequirement.NONE,
    ram_mb: int = 16,
) -> PluginManifest:
    return PluginManifest(
        schema_version=1,
        package=PackageMetadata(
            id=package_id,
            name=package_id,
            version="1.0.0",
            publisher="Tests",
            license="GPL-3.0-or-later",
        ),
        runtime=RuntimeDefinition(
            language="python",
            python=">=3.14",
            entrypoint="tests.runtime:create_runtime",
            isolation=isolation,
            load_policy=LoadPolicy.LOAD_ON_EXECUTION,
            unload_policy=UnloadPolicy.UNLOAD_AFTER_EXECUTION,
            idle_timeout_seconds=0,
        ),
        resources=ResourceRequirements(
            gpu=gpu,
            estimated_ram_mb=ram_mb,
            estimated_vram_mb=0,
            network=False,
        ),
        nodes=tuple(definitions),
        source_path=Path(f"/tmp/{package_id}.ng-node.toml"),
    )


def _catalogue(*manifests: PluginManifest) -> NodeCatalogue:
    return NodeCatalogue(
        manifests=tuple(manifests),
        nodes=tuple(node for manifest in manifests for node in manifest.nodes),
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


def test_compiler_fuses_explicit_compatible_linear_nodes_with_traceability() -> None:
    traits = NodeExecutionTraits(
        pure=True,
        deterministic=True,
        cache_policy="content",
        fusion_kind="integer_transform",
    )
    source_definition = _definition(
        "ng.fusion.source",
        inputs=("value",),
        execution=traits,
    )
    target_definition = _definition(
        "ng.fusion.target",
        inputs=("value",),
        execution=traits,
    )
    catalogue = _catalogue(_manifest(source_definition, target_definition))
    source = _node(1, source_definition, inputs={"value": 1})
    target = _node(2, target_definition)
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(source, target),
        edges=(Edge(source.id, "value", target.id, "value"),),
        outputs={"result": OutputBinding(target.id, "value")},
    )

    plan = GraphCompiler.compile(graph, CompileContext(catalogue=catalogue))

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.member_node_ids == (source.id, target.id)
    assert step.fusion is not None
    assert step.fusion.kind == "integer_transform"
    assert step.fusion.member_node_ids == step.member_node_ids
    assert plan.fusions == (step.fusion,)
    assert plan.step_for_node(source.id) is step
    assert plan.step_for_node(target.id) is step


@pytest.mark.parametrize(
    ("mutation", "expected_steps"),
    (
        ("fanout", 3),
        ("different_kind", 2),
        ("side_effect", 2),
        ("different_runtime", 2),
        ("different_resources", 2),
        ("schema_mismatch", 2),
    ),
)
def test_compiler_never_fuses_without_complete_compatibility_evidence(
    mutation: str,
    expected_steps: int,
) -> None:
    safe = NodeExecutionTraits(
        pure=True,
        deterministic=True,
        cache_policy="content",
        fusion_kind="integer_transform",
    )
    target_traits = safe
    if mutation == "different_kind":
        target_traits = NodeExecutionTraits(
            pure=True,
            deterministic=True,
            cache_policy="content",
            fusion_kind="other_transform",
        )
    elif mutation == "side_effect":
        target_traits = NodeExecutionTraits(
            pure=False,
            deterministic=True,
            cache_policy="never",
            fusion_kind=None,
            side_effects=("filesystem",),
        )
    source_definition = _definition(
        "ng.fusion.guard_source",
        inputs=("value",),
        execution=safe,
    )
    target_definition = _definition(
        "ng.fusion.guard_target",
        inputs=("value",),
        execution=target_traits,
        package_id=(
            "org.comfyng.other"
            if mutation == "different_runtime"
            else "org.comfyng.tests"
        ),
        input_port_schema=(
            {"type": "number"} if mutation == "schema_mismatch" else None
        ),
    )
    source_manifest = _manifest(source_definition)
    if mutation in {"different_runtime", "different_resources"}:
        target_manifest = _manifest(
            target_definition,
            package_id=target_definition.package_id,
            ram_mb=32 if mutation == "different_resources" else 16,
        )
        catalogue = _catalogue(source_manifest, target_manifest)
    else:
        catalogue = _catalogue(_manifest(source_definition, target_definition))
    source = _node(1, source_definition, inputs={"value": 1})
    target = _node(2, target_definition)
    nodes = [source, target]
    edges = [Edge(source.id, "value", target.id, "value")]
    outputs = {"result": OutputBinding(target.id, "value")}
    if mutation == "fanout":
        extra = _node(3, target_definition)
        nodes.append(extra)
        edges.append(Edge(source.id, "value", extra.id, "value"))
        outputs["extra"] = OutputBinding(extra.id, "value")

    plan = GraphCompiler.compile(
        Graph(
            id=uuid4(),
            version=1,
            nodes=tuple(nodes),
            edges=tuple(edges),
            outputs=outputs,
        ),
        CompileContext(catalogue=catalogue),
    )

    assert len(plan.steps) == expected_steps
    assert plan.fusions == ()


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


def test_cache_reuse_requires_explicit_safe_execution_traits() -> None:
    safe = _definition(
        "ng.test.safe_cache",
        inputs=("value",),
        execution=NodeExecutionTraits(
            pure=True,
            deterministic=True,
            cache_policy="content",
        ),
    )
    unsafe = _node(1, SOURCE, inputs={"value": 1})
    safe_node = _node(2, safe, inputs={"value": 1})

    unsafe_plan = GraphCompiler.compile(
        Graph(id=uuid4(), version=1, nodes=(unsafe,), edges=()),
        TEST_CONTEXT,
    )
    safe_plan = GraphCompiler.compile(
        Graph(id=uuid4(), version=1, nodes=(safe_node,), edges=()),
        CompileContext(
            catalogue=NodeCatalogue(manifests=(), nodes=(safe,)),
        ),
    )

    assert unsafe_plan.steps[0].cacheable is False
    assert safe_plan.steps[0].cacheable is True


def test_control_nodes_are_never_reusable_cache_entries() -> None:
    catalogue = NodeCatalogue.discover()
    switch = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.switch",
        type_version="1.0.0",
        inputs={"condition": True, "true_value": 1, "false_value": 0},
    )
    graph = Graph(id=uuid4(), version=1, nodes=(switch,), edges=())

    plan = GraphCompiler.compile(graph, CompileContext(catalogue=catalogue))

    assert plan.steps[0].is_constant is True
    assert plan.steps[0].cacheable is False


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


def test_preflight_rejects_duplicate_parent_subgraph_call_ids() -> None:
    inner = _node(10, IDENTITY)
    subgraph = Graph(
        id=UUID(int=500),
        version=1,
        nodes=(inner,),
        edges=(),
        inputs={"input": InputBinding(inner.id, "value")},
        outputs={"output": OutputBinding(inner.id, "value")},
    )
    first = NodeInstance(
        id=UUID(int=2),
        type_id="ng.subgraph.identity",
        type_version="1.0.0",
        inputs={"input": 1},
    )
    duplicate = NodeInstance(
        id=first.id,
        type_id=first.type_id,
        type_version=first.type_version,
        inputs={"input": 2},
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(first, duplicate),
        edges=(),
        outputs={"result": OutputBinding(first.id, "output")},
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            graph,
            CompileContext(
                catalogue=TEST_CATALOGUE,
                subgraphs={(first.type_id, first.type_version): subgraph},
            ),
        )

    assert "duplicate_node_id" in {item.code for item in captured.value.diagnostics}


def test_preflight_rejects_literal_boundary_bound_to_missing_child_node() -> None:
    missing_id = UUID(int=999)
    subgraph = Graph(
        id=UUID(int=500),
        version=1,
        nodes=(),
        edges=(),
        inputs={"input": InputBinding(missing_id, "value")},
    )
    call = NodeInstance(
        id=UUID(int=2),
        type_id="ng.subgraph.missing_target",
        type_version="1.0.0",
        inputs={"input": 1},
    )
    graph = Graph(id=uuid4(), version=1, nodes=(call,), edges=())

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            graph,
            CompileContext(
                catalogue=TEST_CATALOGUE,
                subgraphs={(call.type_id, call.type_version): subgraph},
            ),
        )

    assert "missing_graph_input_node" in {
        item.code for item in captured.value.diagnostics
    }


def test_preflight_recursively_validates_unused_registered_subgraphs() -> None:
    source = _node(1, SOURCE, inputs={"value": 1})
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(source,),
        edges=(),
        outputs={"result": OutputBinding(source.id, "value")},
    )
    invalid_child = Graph(
        id=UUID(int=500),
        version=1,
        nodes=(),
        edges=(Edge(UUID(int=1), "value", UUID(int=2), "value"),),
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            graph,
            CompileContext(
                catalogue=TEST_CATALOGUE,
                subgraphs={
                    ("ng.subgraph.unused", "1.0.0"): invalid_child,
                },
            ),
        )

    assert {item.code for item in captured.value.diagnostics}.issuperset(
        {"missing_source_node", "missing_target_node"}
    )


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


def test_condition_region_contains_exclusive_typed_branches() -> None:
    catalogue = NodeCatalogue.discover()
    true_source = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.route",
        type_version="1.0.0",
        inputs={"value": 10, "route": "true"},
    )
    false_source = NodeInstance(
        id=UUID(int=2),
        type_id="ng.control.route",
        type_version="1.0.0",
        inputs={"value": 20, "route": "false"},
    )
    switch = NodeInstance(
        id=UUID(int=3),
        type_id="ng.control.switch",
        type_version="1.0.0",
        inputs={"condition": True},
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(switch, false_source, true_source),
        edges=(
            Edge(true_source.id, "value", switch.id, "true_value"),
            Edge(false_source.id, "value", switch.id, "false_value"),
        ),
        outputs={"result": OutputBinding(switch.id, "value")},
    )

    plan = GraphCompiler.compile(graph, CompileContext(catalogue=catalogue))

    assert len(plan.control_regions) == 1
    region = plan.control_regions[0]
    assert isinstance(region, ControlRegion)
    assert region.kind == "condition"
    assert region.controller_step_id == switch.id
    assert region.parent_controller_step_id is None
    assert region.nesting_depth == 0
    assert tuple(branch.name for branch in region.branches) == (
        "false_value",
        "true_value",
    )
    members = {branch.name: branch.member_step_ids for branch in region.branches}
    assert members == {
        "false_value": (false_source.id,),
        "true_value": (true_source.id,),
    }
    assert all(isinstance(branch, ControlBranch) for branch in region.branches)


def test_loop_region_identifies_iterable_body_boundary_and_bound() -> None:
    catalogue = NodeCatalogue.discover()
    loop = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.for_each",
        type_version="1.0.0",
        inputs={"items": [1, 2, 3]},
    )
    body = NodeInstance(
        id=UUID(int=2),
        type_id="ng.control.route",
        type_version="1.0.0",
        inputs={"route": "body"},
    )
    collect = NodeInstance(
        id=UUID(int=3),
        type_id="ng.control.collect",
        type_version="1.0.0",
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(collect, body, loop),
        edges=(
            Edge(loop.id, "item", body.id, "value"),
            Edge(body.id, "value", collect.id, "item"),
            Edge(loop.id, "index", collect.id, "index"),
        ),
        outputs={"items": OutputBinding(collect.id, "items")},
    )

    plan = GraphCompiler.compile(
        graph,
        CompileContext(catalogue=catalogue, max_loop_iterations=3),
    )

    region = plan.control_regions[0]
    assert region.kind == "loop"
    assert region.controller_step_id == loop.id
    assert region.max_iterations == 3
    assert region.iterable_step_ids == ()
    assert region.body_step_ids == (body.id,)
    assert region.boundary_step_ids == (collect.id,)


def test_nested_condition_region_is_owned_by_enclosing_loop() -> None:
    catalogue = NodeCatalogue.discover()
    loop = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.for_each",
        type_version="1.0.0",
        inputs={"items": [True, False]},
    )
    switch = NodeInstance(
        id=UUID(int=2),
        type_id="ng.control.switch",
        type_version="1.0.0",
        inputs={"condition": True, "false_value": False},
    )
    collect = NodeInstance(
        id=UUID(int=3),
        type_id="ng.control.collect",
        type_version="1.0.0",
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(loop, switch, collect),
        edges=(
            Edge(loop.id, "item", switch.id, "true_value"),
            Edge(switch.id, "value", collect.id, "item"),
            Edge(loop.id, "index", collect.id, "index"),
        ),
        outputs={"items": OutputBinding(collect.id, "items")},
    )

    plan = GraphCompiler.compile(
        graph,
        CompileContext(catalogue=catalogue, max_loop_iterations=2),
    )
    regions = {region.controller_step_id: region for region in plan.control_regions}

    assert regions[loop.id].parent_controller_step_id is None
    assert regions[loop.id].nesting_depth == 0
    assert regions[switch.id].parent_controller_step_id == loop.id
    assert regions[switch.id].nesting_depth == 1
    assert loop.id not in {
        member
        for branch in regions[switch.id].branches
        for member in branch.member_step_ids
    }


def test_nontrivial_loop_without_collect_boundary_is_rejected() -> None:
    catalogue = NodeCatalogue.discover()
    loop = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.for_each",
        type_version="1.0.0",
        inputs={"items": [1]},
    )
    body = NodeInstance(
        id=UUID(int=2),
        type_id="ng.control.route",
        type_version="1.0.0",
        inputs={"route": "body"},
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(loop, body),
        edges=(Edge(loop.id, "item", body.id, "value"),),
        outputs={"result": OutputBinding(body.id, "value")},
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            graph,
            CompileContext(catalogue=catalogue, max_loop_iterations=1),
        )

    assert "malformed_loop_region" in {
        diagnostic.code for diagnostic in captured.value.diagnostics
    }


def test_subgraph_function_and_macro_expansion_preserve_nested_trace() -> None:
    leaf = _node(30, IDENTITY, inputs={"value": 7})
    macro = Graph(
        id=UUID(int=700),
        version=1,
        nodes=(leaf,),
        edges=(),
        outputs={"output": OutputBinding(leaf.id, "value")},
    )
    macro_call = NodeInstance(
        id=UUID(int=20),
        type_id="ng.macro.identity",
        type_version="1.0.0",
        metadata={"construct_kind": "macro"},
    )
    function = Graph(
        id=UUID(int=600),
        version=1,
        nodes=(macro_call,),
        edges=(),
        outputs={"output": OutputBinding(macro_call.id, "output")},
    )
    function_call = NodeInstance(
        id=UUID(int=10),
        type_id="ng.function.identity",
        type_version="1.0.0",
        metadata={"construct_kind": "function"},
    )
    root = Graph(
        id=UUID(int=500),
        version=1,
        nodes=(function_call,),
        edges=(),
        outputs={"result": OutputBinding(function_call.id, "output")},
    )

    plan = GraphCompiler.compile(
        root,
        CompileContext(
            catalogue=TEST_CATALOGUE,
            subgraphs={
                (function_call.type_id, function_call.type_version): function,
                (macro_call.type_id, macro_call.type_version): macro,
            },
        ),
    )

    assert all(isinstance(trace, SubgraphTrace) for trace in plan.subgraph_traces)
    assert tuple(trace.kind for trace in plan.subgraph_traces) == (
        "function",
        "macro",
    )
    outer, inner = plan.subgraph_traces
    assert outer.parent_call_node_id is None
    assert outer.depth == 0
    assert inner.parent_call_node_id == outer.call_node_id
    assert inner.depth == 1
    assert set(outer.member_node_ids) == set(plan.topological_order)
    assert set(inner.member_node_ids).issubset(outer.member_node_ids)


def test_invalid_subgraph_construct_kind_is_a_structured_error() -> None:
    inner = _node(10, IDENTITY, inputs={"value": 1})
    child = Graph(id=UUID(int=500), version=1, nodes=(inner,), edges=())
    call = NodeInstance(
        id=UUID(int=2),
        type_id="ng.subgraph.invalid_kind",
        type_version="1.0.0",
        metadata={"construct_kind": "procedure"},
    )

    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(
            Graph(id=uuid4(), version=1, nodes=(call,), edges=()),
            CompileContext(
                catalogue=TEST_CATALOGUE,
                subgraphs={(call.type_id, call.type_version): child},
            ),
        )

    assert "invalid_subgraph_kind" in {
        diagnostic.code for diagnostic in captured.value.diagnostics
    }


def test_execution_contracts_reject_inconsistent_dependencies_groups_and_plan() -> None:
    plan = GraphCompiler.compile(_diamond_graph(), TEST_CONTEXT)
    first = plan.steps[0]
    second_group = plan.groups[1]

    with pytest.raises(ValueError, match="dependencies"):
        msgspec.structs.replace(first, dependencies=(first.node_id,))
    with pytest.raises(ValueError, match="group_index"):
        msgspec.structs.replace(second_group, steps=(first,))
    with pytest.raises(ValueError, match="topological_order"):
        msgspec.structs.replace(
            plan,
            topological_order=tuple(reversed(plan.topological_order)),
        )
    with pytest.raises(ValueError, match="groups"):
        msgspec.structs.replace(plan, groups=plan.groups[:-1])


def test_execution_plan_contract_round_trips_with_regions_and_fusions() -> None:
    catalogue = NodeCatalogue.discover()
    switch = NodeInstance(
        id=UUID(int=1),
        type_id="ng.control.switch",
        type_version="1.0.0",
        inputs={"condition": True, "true_value": 1, "false_value": 0},
    )
    plan = GraphCompiler.compile(
        Graph(id=uuid4(), version=1, nodes=(switch,), edges=()),
        CompileContext(catalogue=catalogue),
    )

    restored = ExecutionPlan.from_json(plan.to_json())

    assert restored == plan
    assert restored.control_regions == plan.control_regions


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
