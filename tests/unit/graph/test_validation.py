from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from comfyng.core.enums import SerializationStrategy, TransferPolicy
from comfyng.graph.compiler import CompileContext, GraphCompilationError, GraphCompiler
from comfyng.graph.types import (
    DEFAULT_TYPE_REGISTRY,
    Edge,
    Graph,
    NodeInstance,
    OutputBinding,
    PortTypeDefinition,
    TypeRef,
    TypeRegistry,
)
from comfyng.graph.validation import validate_graph
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import NodeDefinition


def _definition(
    node_id: str,
    *,
    inputs: dict[str, dict[str, object]],
    outputs: dict[str, dict[str, object]],
    required: tuple[str, ...] = (),
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
            "properties": inputs,
            "required": list(required),
            "x-comfyng-optional": sorted(set(inputs) - set(required)),
        },
        output_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": outputs,
            "required": list(outputs),
        },
    )


def _context(
    *definitions: NodeDefinition, registry: TypeRegistry | None = None
) -> CompileContext:
    return CompileContext(
        catalogue=NodeCatalogue(manifests=(), nodes=tuple(definitions)),
        type_registry=registry or DEFAULT_TYPE_REGISTRY,
    )


def _node(
    value: int,
    type_id: str,
    *,
    inputs: dict[str, object] | None = None,
) -> NodeInstance:
    return NodeInstance(
        id=UUID(int=value),
        type_id=type_id,
        type_version="1.0.0",
        inputs=inputs or {},
    )


def test_validation_reports_duplicate_node_ids_and_compilation_rejects_them() -> None:
    definition = _definition(
        "ng.test.source",
        inputs={"value": {"type": "integer"}},
        outputs={"value": {"type": "integer"}},
        required=("value",),
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(
            _node(1, definition.id, inputs={"value": 1}),
            _node(1, definition.id, inputs={"value": 2}),
        ),
        edges=(),
    )

    diagnostics = validate_graph(graph, _context(definition))

    assert [item.code for item in diagnostics if item.severity == "error"] == [
        "duplicate_node_id"
    ]
    with pytest.raises(GraphCompilationError) as captured:
        GraphCompiler.compile(graph, _context(definition))
    assert captured.value.diagnostics == diagnostics


def test_validation_reports_missing_edge_ports() -> None:
    source = _definition(
        "ng.test.source",
        inputs={"value": {"type": "integer"}},
        outputs={"value": {"type": "integer"}},
        required=("value",),
    )
    sink = _definition(
        "ng.test.sink",
        inputs={"value": {"type": "integer"}},
        outputs={"value": {"type": "integer"}},
        required=("value",),
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(
            _node(1, source.id, inputs={"value": 1}),
            _node(2, sink.id),
        ),
        edges=(Edge(UUID(int=1), "missing", UUID(int=2), "missing"),),
    )

    codes = {item.code for item in validate_graph(graph, _context(source, sink))}

    assert "missing_source_port" in codes
    assert "missing_target_port" in codes


def test_validation_rejects_incompatible_type_names_and_versions() -> None:
    model_v1 = {"type": "string", "x-comfyng-type": "NG_MODEL@1"}
    model_v2 = {"type": "string", "x-comfyng-type": "NG_MODEL@2"}
    latent_v1 = {"type": "string", "x-comfyng-type": "NG_LATENT@1"}
    source = _definition("ng.test.source", inputs={}, outputs={"value": model_v1})
    wrong_name = _definition(
        "ng.test.wrong_name",
        inputs={"value": latent_v1},
        outputs={"value": latent_v1},
        required=("value",),
    )
    wrong_version = _definition(
        "ng.test.wrong_version",
        inputs={"value": model_v2},
        outputs={"value": model_v2},
        required=("value",),
    )
    registry = TypeRegistry(DEFAULT_TYPE_REGISTRY.definitions)
    registry.register(
        PortTypeDefinition(
            ref=TypeRef("NG_MODEL", 2),
            schema={"type": "string"},
            serialization_strategy=SerializationStrategy.JSON,
            transfer_policy=TransferPolicy.INLINE,
        )
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(
            _node(1, source.id),
            _node(2, wrong_name.id),
            _node(3, wrong_version.id),
        ),
        edges=(
            Edge(UUID(int=1), "value", UUID(int=2), "value"),
            Edge(UUID(int=1), "value", UUID(int=3), "value"),
        ),
    )

    codes = {
        item.code
        for item in validate_graph(
            graph,
            _context(source, wrong_name, wrong_version, registry=registry),
        )
    }

    assert "incompatible_types" in codes
    assert "type_version_mismatch" in codes


def test_validation_rejects_missing_required_inputs_and_invalid_literals() -> None:
    sink = _definition(
        "ng.test.sink",
        inputs={"value": {"type": "integer", "minimum": 1}},
        outputs={"value": {"type": "integer"}},
        required=("value",),
    )
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(
            _node(1, sink.id),
            _node(2, sink.id, inputs={"value": 0}),
        ),
        edges=(),
    )

    codes = [item.code for item in validate_graph(graph, _context(sink))]

    assert "missing_required_input" in codes
    assert "invalid_literal_input" in codes


def test_validation_reports_unused_outputs_without_rejecting_the_plan() -> None:
    source = _definition(
        "ng.test.source",
        inputs={"value": {"type": "integer"}},
        outputs={
            "used": {"type": "integer"},
            "unused": {"type": "integer"},
        },
        required=("value",),
    )
    node = _node(1, source.id, inputs={"value": 1})
    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(node,),
        edges=(),
        outputs={"result": OutputBinding(node.id, "used")},
    )

    plan = GraphCompiler.compile(graph, _context(source))

    assert [item.code for item in plan.diagnostics] == ["unused_output"]
    assert plan.diagnostics[0].node_id == node.id
    assert plan.diagnostics[0].port == "unused"
