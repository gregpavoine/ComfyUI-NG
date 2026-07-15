from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from comfyng.core.enums import LifecycleState, TransferPolicy
from comfyng.core.errors import (
    DuplicateTypeDefinitionError,
    InvalidLifecycleTransition,
    UnknownTypeDefinitionError,
)
from comfyng.graph.types import (
    Edge,
    Graph,
    NodeInstance,
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

    with pytest.raises(AttributeError):
        capabilities.family = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        handle.size_bytes = 0  # type: ignore[misc]


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
                inputs={"seed": 7},
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
        inputs={"prompt": {"node_id": str(second_id), "port": "prompt"}},
        outputs={"latent": {"node_id": str(second_id), "port": "latent"}},
    )

    assert TensorHandle.from_json(tensor.to_json()) == tensor
    assert Graph.from_json(graph.to_json()) == graph
    assert Graph.from_builtins(graph.to_builtins()) == graph

    with pytest.raises(AttributeError):
        tensor.owner_worker = "gpu-1"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        graph.version = 4  # type: ignore[misc]


def test_versioned_type_registry_rejects_duplicates_and_unknown_versions() -> None:
    registry = TypeRegistry()
    image_v1 = PortTypeDefinition(
        ref=TypeRef(name="NG_IMAGE", version=1),
        schema={"type": "string", "format": "comfyng-handle"},
        transfer_policy=TransferPolicy.HANDLE,
    )
    registry.register(image_v1)

    assert registry.resolve("NG_IMAGE", 1) == image_v1
    assert registry.resolve_ref("NG_IMAGE@1") == image_v1

    with pytest.raises(DuplicateTypeDefinitionError):
        registry.register(image_v1)
    with pytest.raises(UnknownTypeDefinitionError):
        registry.resolve("NG_IMAGE", 2)
    with pytest.raises(ValueError):
        TypeRef.parse("NG_IMAGE")


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
