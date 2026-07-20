from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from comfyng.core.enums import (
    LifecycleState,
    SerializationStrategy,
    TransferPolicy,
)
from comfyng.core.errors import (
    DuplicateTypeDefinitionError,
    InvalidLifecycleTransition,
    JsonValueValidationError,
    UnknownTypeDefinitionError,
)
from comfyng.graph.types import (
    Edge,
    Graph,
    InputBinding,
    NodeInstance,
    OutputBinding,
    PortTypeDefinition,
    TensorHandle,
    TypeRef,
    TypeRegistry,
)
from comfyng.models.capabilities import ModelCapabilities, ModelHandle


def _capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        family="flux1",
        architecture="flux-transformer-2d",
        task_types=frozenset(("text-to-image", "image-to-image")),
        latent_channels=16,
        latent_scale_factor=8,
        prediction_type="flow_matching",
        supported_dtypes=("bfloat16", "float16"),
        supported_quantizations=("none", "int8"),
        text_encoder_layout=("clip_l", "t5xxl"),
        supports_negative_prompt=False,
        supports_cfg=False,
        supports_embedded_guidance=True,
        supports_img2img=True,
        supports_inpainting=False,
        supports_lora=True,
        supports_control=True,
        samplers=("euler",),
        schedulers=("simple", "beta"),
        attention_backends=("sdpa", "flash_attention_2"),
    )


def _capability_constructor_values() -> dict[str, object]:
    capabilities = _capabilities()
    values = capabilities.to_builtins()
    values["task_types"] = capabilities.task_types
    for field in (
        "supported_dtypes",
        "supported_quantizations",
        "text_encoder_layout",
        "samplers",
        "schedulers",
        "attention_backends",
    ):
        values[field] = getattr(capabilities, field)
    return values


def test_capabilities_and_model_handle_are_frozen_versioned_contracts(
    tmp_path: Path,
) -> None:
    capabilities = _capabilities()
    handle = ModelHandle(
        id=uuid4(),
        family=capabilities.family,
        architecture=capabilities.architecture,
        local_path=(tmp_path / "flux.safetensors").resolve(),
        sha256="a" * 64,
        size_bytes=42,
        source_provider="local",
        source_model_id=None,
        source_revision=None,
        metadata={"format": "safetensors", "shards": 1},
    )

    assert ModelCapabilities.from_json(capabilities.to_json()) == capabilities
    assert ModelHandle.from_json(handle.to_json()) == handle
    assert capabilities.contract_type == "comfyng.model-capabilities"
    assert capabilities.contract_version == 1

@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("family", ""),
        ("latent_channels", 0),
        ("latent_scale_factor", -1),
        ("supported_dtypes", ("float16", "float16")),
        ("samplers", ()),
        ("schedulers", ()),
    ),
)
def test_model_capabilities_reject_invalid_values(field: str, value: object) -> None:
    values = _capabilities().to_builtins()
    values[field] = value

    with pytest.raises(ValueError):
        ModelCapabilities.from_builtins(values)


def test_model_capabilities_requires_a_frozenset_for_task_types() -> None:
    values = _capability_constructor_values()
    values["task_types"] = tuple(values["task_types"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="task_types"):
        ModelCapabilities(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    (
        "supported_dtypes",
        "supported_quantizations",
        "text_encoder_layout",
        "samplers",
        "schedulers",
        "attention_backends",
    ),
)
def test_model_capabilities_requires_tuples_for_ordered_collections(field: str) -> None:
    values = _capability_constructor_values()
    values[field] = frozenset(values[field])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=field):
        ModelCapabilities(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("family", chr(0xD800)),
        ("task_types", frozenset(("text-to-image", chr(0xD800)))),
        ("supported_dtypes", ("float16", chr(0xD800))),
    ),
)
def test_model_capabilities_rejects_unsafe_unicode_strings(
    field: str,
    value: object,
) -> None:
    values = _capability_constructor_values()
    values[field] = value

    with pytest.raises(ValueError, match="Unicode"):
        ModelCapabilities(**values)  # type: ignore[arg-type]


def test_model_handle_rejects_relative_paths_invalid_hashes_and_negative_sizes(
    tmp_path: Path,
) -> None:
    common = dict(
        id=uuid4(),
        family="flux1",
        architecture="flux-transformer-2d",
        local_path=(tmp_path / "model.safetensors").resolve(),
        sha256="b" * 64,
        size_bytes=1,
        source_provider=None,
        source_model_id=None,
        source_revision=None,
        metadata={},
    )

    with pytest.raises(ValueError, match="absolute"):
        ModelHandle(**(common | {"local_path": Path("relative/model.safetensors")}))
    with pytest.raises(ValueError, match="sha256"):
        ModelHandle(**(common | {"sha256": "not-a-hash"}))
    with pytest.raises(ValueError, match="size_bytes"):
        ModelHandle(**(common | {"size_bytes": -1}))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("family", chr(0xD800)),
        ("source_provider", chr(0xD800)),
        ("local_path", Path("/") / chr(0xD800)),
    ),
)
def test_model_handle_rejects_unsafe_unicode_strings(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    values = dict(
        id=uuid4(),
        family="flux1",
        architecture="flux-transformer-2d",
        local_path=(tmp_path / "model.safetensors").resolve(),
        sha256="d" * 64,
        size_bytes=1,
        source_provider=None,
        source_model_id=None,
        source_revision=None,
        metadata={},
    )
    values[field] = value

    with pytest.raises(ValueError, match="Unicode"):
        ModelHandle(**values)


def test_tensor_handle_rejects_unsafe_unicode_strings() -> None:
    with pytest.raises(ValueError, match="Unicode"):
        TensorHandle(
            id=uuid4(),
            storage=chr(0xD800),
            shape=(1,),
            dtype="float16",
            device="cpu",
            owner_worker="worker-0",
            byte_size=2,
        )


def test_graph_and_tensor_handles_round_trip_without_payload_copies() -> None:
    first_id = uuid4()
    second_id = uuid4()
    tensor = TensorHandle(
        id=uuid4(),
        storage="shared_memory",
        shape=(1, 16, 128, 128),
        dtype="float16",
        device="cuda:0",
        owner_worker="gpu-0",
        byte_size=524_288,
    )
    graph = Graph(
        id=uuid4(),
        version=3,
        nodes=(
            NodeInstance(
                id=first_id,
                type_id="ng.latent.empty",
                type_version="1.0.0",
                inputs={"width": 1024, "height": 1024},
            ),
            NodeInstance(
                id=second_id,
                type_id="ng.sample.run",
                type_version="1.0.0",
                inputs={
                    "seed": 7,
                    "options": [1, {"preview": True}, None, 0.5],
                },
            ),
        ),
        edges=(
            Edge(
                source_node_id=first_id,
                source_port="latent",
                target_node_id=second_id,
                target_port="latent",
            ),
        ),
        inputs={
            "prompt": InputBinding(node_id=second_id, port="prompt"),
        },
        outputs={
            "latent": OutputBinding(node_id=second_id, port="latent"),
        },
    )

    assert TensorHandle.from_json(tensor.to_json()) == tensor
    assert Graph.from_json(graph.to_json()) == graph
    assert Graph.from_builtins(graph.to_builtins()) == graph


@pytest.mark.parametrize(
    "invalid",
    (
        object(),
        b"bytes-are-not-json",
        {1: "non-string-key"},
        float("nan"),
        float("inf"),
        chr(0xD800),
    ),
    ids=(
        "object",
        "bytes",
        "non-string-key",
        "nan",
        "infinity",
        "unpaired-surrogate",
    ),
)
def test_node_instances_reject_values_without_stable_json_round_trips(
    invalid: object,
) -> None:
    with pytest.raises(JsonValueValidationError) as caught:
        NodeInstance(
            id=uuid4(),
            type_id="ng.test.value",
            type_version="1.0.0",
            inputs={"payload": invalid},
        )

    assert caught.value.path == "$.inputs.payload"


def test_json_contracts_deep_freeze_caller_values_and_round_trip() -> None:
    source = {
        "payload": {
            "items": [1, {"label": "original"}],
            "coordinates": (2, 3),
        }
    }
    node = NodeInstance(
        id=uuid4(),
        type_id="ng.test.value",
        type_version="1.0.0",
        inputs=source,
    )

    source["payload"]["items"][1]["label"] = "changed"  # type: ignore[index]
    source["payload"]["items"].append(4)  # type: ignore[union-attr]

    assert type(node.inputs).__name__ == "FrozenDict"
    assert node.inputs["payload"]["items"] == (1, {"label": "original"})
    assert node.inputs["payload"]["coordinates"] == (2, 3)
    with pytest.raises(TypeError, match="immutable"):
        node.inputs["extra"] = True  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        node.inputs["payload"]["items"][1]["label"] = "changed"  # type: ignore[index]

    encoded = node.to_json()
    decoded = NodeInstance.from_json(encoded)
    assert decoded == node
    assert type(decoded.inputs).__name__ == "FrozenDict"
    assert decoded.to_json() == encoded


def test_graph_binding_maps_are_copied_and_immutable() -> None:
    node_id = uuid4()
    input_binding = InputBinding(node_id=node_id, port="prompt")
    output_binding = OutputBinding(node_id=node_id, port="image")
    inputs = {"prompt": input_binding}
    outputs = {"image": output_binding}

    graph = Graph(
        id=uuid4(),
        version=1,
        nodes=(),
        edges=(),
        inputs=inputs,
        outputs=outputs,
    )
    inputs["other"] = input_binding
    outputs.clear()

    assert graph.inputs == {"prompt": input_binding}
    assert graph.outputs == {"image": output_binding}
    assert type(graph.inputs).__name__ == "FrozenDict"
    with pytest.raises(TypeError, match="immutable"):
        graph.inputs["other"] = input_binding  # type: ignore[index]
    assert Graph.from_json(graph.to_json()) == graph


def test_port_schemas_are_deeply_frozen() -> None:
    schema = {
        "type": "object",
        "properties": {"values": {"enum": ["one", "two"]}},
    }
    definition = PortTypeDefinition(
        ref=TypeRef(name="NG_TEST_VALUE", version=1),
        schema=schema,
        serialization_strategy=SerializationStrategy.JSON,
        transfer_policy=TransferPolicy.INLINE,
    )
    schema["properties"]["values"]["enum"].append("three")  # type: ignore[index]

    assert definition.schema["properties"]["values"]["enum"] == ("one", "two")
    with pytest.raises(TypeError, match="immutable"):
        definition.schema["properties"]["extra"] = {}  # type: ignore[index]
    assert PortTypeDefinition.from_json(definition.to_json()) == definition


def test_model_metadata_and_graph_bindings_are_strictly_typed(tmp_path: Path) -> None:
    with pytest.raises(JsonValueValidationError) as caught:
        ModelHandle(
            id=uuid4(),
            family="flux1",
            architecture="flux-transformer-2d",
            local_path=(tmp_path / "model.safetensors").resolve(),
            sha256="c" * 64,
            size_bytes=1,
            source_provider=None,
            source_model_id=None,
            source_revision=None,
            metadata={"unsafe": object()},
        )
    assert caught.value.path == "$.metadata.unsafe"

    node_id = uuid4()
    with pytest.raises(ValueError, match="InputBinding"):
        Graph(
            id=uuid4(),
            version=1,
            nodes=(),
            edges=(),
            inputs={"prompt": {"node_id": str(node_id), "port": "prompt"}},
            outputs={},
        )


def test_versioned_type_registry_rejects_duplicates_and_unknown_versions() -> None:
    registry = TypeRegistry()
    image_v1 = PortTypeDefinition(
        ref=TypeRef(name="NG_IMAGE", version=1),
        schema={"type": "string", "format": "comfyng-handle"},
        serialization_strategy=SerializationStrategy.SHARED_HANDLE,
        transfer_policy=TransferPolicy.HANDLE,
    )
    registry.register(image_v1)

    assert registry.resolve("NG_IMAGE", 1) == image_v1
    assert registry.resolve_ref("NG_IMAGE@1") == image_v1
    assert image_v1.serialization_strategy is SerializationStrategy.SHARED_HANDLE
    assert PortTypeDefinition.from_json(image_v1.to_json()) == image_v1

    with pytest.raises(DuplicateTypeDefinitionError):
        registry.register(image_v1)
    with pytest.raises(UnknownTypeDefinitionError):
        registry.resolve("NG_IMAGE", 2)
    with pytest.raises(ValueError):
        TypeRef.parse("NG_IMAGE")
    with pytest.raises(ValueError, match="NG_"):
        TypeRef.parse("MODEL@1")


@pytest.mark.parametrize(
    "name",
    ("NG_MODEL", "NG_MODEL_INFO", "NG_A1_2B"),
)
def test_type_names_accept_non_empty_uppercase_segments(name: str) -> None:
    assert TypeRef(name=name, version=1).name == name


@pytest.mark.parametrize(
    "name",
    ("NG__MODEL", "NG_MODEL_", "NG_MODEL__INFO", "NG_", "MODEL"),
)
def test_type_names_reject_empty_or_missing_ng_segments(name: str) -> None:
    with pytest.raises(ValueError, match="NG_"):
        TypeRef(name=name, version=1)


def test_lifecycle_allows_only_declared_transitions() -> None:
    state = LifecycleState.DISCOVERED
    for target in (
        LifecycleState.RESOLVED,
        LifecycleState.PRELOADING,
        LifecycleState.LOADED,
        LifecycleState.READY,
        LifecycleState.BUSY,
        LifecycleState.IDLE,
        LifecycleState.EVICTING,
        LifecycleState.UNLOADED,
    ):
        state = state.transition_to(target)
    assert state is LifecycleState.UNLOADED

    with pytest.raises(InvalidLifecycleTransition):
        LifecycleState.DISCOVERED.transition_to(LifecycleState.BUSY)
    with pytest.raises(InvalidLifecycleTransition):
        LifecycleState.UNLOADED.transition_to(LifecycleState.BUSY)
