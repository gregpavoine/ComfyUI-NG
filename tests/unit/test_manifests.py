from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from comfyng.core.enums import LoadPolicy, UnloadPolicy
from comfyng.core.errors import ManifestValidationError, PathContainmentError
from comfyng.plugins.manifest import (
    NodeExecutionTraits,
    PackageMetadata,
    PluginManifest,
)


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}


def _write_schema(path: Path, schema: object = SCHEMA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema), encoding="utf-8")


def _manifest_text(
    *,
    package_id: str = "org.comfyng.test",
    version: str = "1.2.3",
    node_id: str = "ng.test.echo",
    input_schema: str = "schemas/input.json",
    output_schema: str = "schemas/output.json",
    schema_version: int = 1,
    entrypoint: str = "sentinel_runtime:create_runtime",
    extra: str = "",
) -> str:
    return f'''schema_version = {schema_version}

[package]
id = "{package_id}"
name = "Test Runtime"
version = "{version}"
publisher = "ComfyUI-NG Tests"
license = "GPL-3.0-or-later"

[runtime]
language = "python"
python = ">=3.14"
entrypoint = "{entrypoint}"
isolation = "gpu_model_worker"
load_policy = "LOAD_ON_EXECUTION"
unload_policy = "UNLOAD_AFTER_IDLE"
idle_timeout_seconds = 30

[resources]
gpu = "optional"
estimated_ram_mb = 64
estimated_vram_mb = 0
network = false

[[nodes]]
id = "{node_id}"
display_name = "Test Echo"
input_schema = "{input_schema}"
output_schema = "{output_schema}"
{extra}'''


def _write_manifest(root: Path, **changes: object) -> Path:
    _write_schema(root / "schemas/input.json")
    _write_schema(root / "schemas/output.json")
    path = root / "ng-node.toml"
    path.write_text(_manifest_text(**changes), encoding="utf-8")
    return path


def test_manifest_loads_schemas_and_round_trips_as_a_frozen_contract(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)

    manifest = PluginManifest.load(manifest_path, root=tmp_path)

    assert manifest.package.id == "org.comfyng.test"
    assert manifest.package.version == "1.2.3"
    assert manifest.runtime.entrypoint == "sentinel_runtime:create_runtime"
    assert manifest.nodes[0].version == "1.2.3"
    assert manifest.nodes[0].input_schema == SCHEMA
    assert (
        manifest.nodes[0].input_schema_path
        == (tmp_path / "schemas/input.json").resolve()
    )
    assert PluginManifest.from_json(manifest.to_json()) == manifest
    with pytest.raises(AttributeError):
        manifest.schema_version = 2  # type: ignore[misc]


def test_node_execution_traits_default_to_no_cache_or_fusion() -> None:
    traits = NodeExecutionTraits()

    assert traits.pure is False
    assert traits.deterministic is False
    assert traits.cache_policy == "never"
    assert traits.fusion_kind is None
    assert traits.side_effects == ()


def test_manifest_loads_explicit_safe_node_execution_traits(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        extra="""
[nodes.execution]
pure = true
deterministic = true
cache_policy = "content"
fusion_kind = "pixel_transform"
side_effects = []
""",
    )

    node = PluginManifest.load(manifest_path, root=tmp_path).nodes[0]

    assert node.execution == NodeExecutionTraits(
        pure=True,
        deterministic=True,
        cache_policy="content",
        fusion_kind="pixel_transform",
        side_effects=(),
    )


@pytest.mark.parametrize(
    "execution",
    (
        """
[nodes.execution]
pure = false
deterministic = true
cache_policy = "content"
""",
        """
[nodes.execution]
pure = true
deterministic = true
cache_policy = "content"
side_effects = ["filesystem_write"]
""",
    ),
)
def test_manifest_rejects_unsafe_cache_execution_traits(
    tmp_path: Path,
    execution: str,
) -> None:
    manifest_path = _write_manifest(tmp_path, extra=execution)

    with pytest.raises(ManifestValidationError, match="execution"):
        PluginManifest.load(manifest_path, root=tmp_path)


def test_manifest_direct_construction_canonicalizes_mutable_metadata(
    tmp_path: Path,
) -> None:
    loaded = PluginManifest.load(_write_manifest(tmp_path), root=tmp_path)
    permission_source = {"network": True, "filesystem": False}
    dependency_source = ["org.comfyng.runtime", "org.comfyng.models"]

    manifest = PluginManifest(
        schema_version=loaded.schema_version,
        package=loaded.package,
        runtime=loaded.runtime,
        resources=loaded.resources,
        nodes=loaded.nodes,
        source_path=loaded.source_path,
        permissions=MappingProxyType(permission_source),
        dependencies=dependency_source,  # type: ignore[arg-type]
        signature=None,
    )
    permission_source["network"] = False
    dependency_source.append("org.comfyng.late-mutation")

    assert type(manifest.permissions).__name__ == "FrozenDict"
    assert manifest.permissions == {"network": True, "filesystem": False}
    assert manifest.dependencies == (
        "org.comfyng.runtime",
        "org.comfyng.models",
    )
    with pytest.raises(TypeError, match="immutable"):
        manifest.permissions["network"] = False  # type: ignore[index]
    assert PluginManifest.from_json(manifest.to_json()) == manifest


@pytest.mark.parametrize(
    ("permissions", "dependencies"),
    (
        ({"network": {"nested": True}}, ()),
        ({object(): True}, ()),
        ({}, (object(),)),
        ({}, (chr(0xD800),)),
        ({chr(0xD800): True}, ()),
    ),
)
def test_manifest_direct_construction_rejects_unsupported_metadata(
    tmp_path: Path,
    permissions: object,
    dependencies: object,
) -> None:
    loaded = PluginManifest.load(_write_manifest(tmp_path), root=tmp_path)

    with pytest.raises(ValueError):
        PluginManifest(
            schema_version=loaded.schema_version,
            package=loaded.package,
            runtime=loaded.runtime,
            resources=loaded.resources,
            nodes=loaded.nodes,
            source_path=loaded.source_path,
            permissions=permissions,  # type: ignore[arg-type]
            dependencies=dependencies,  # type: ignore[arg-type]
            signature=None,
        )


def test_plugin_package_rejects_unsafe_unicode_strings() -> None:
    with pytest.raises(ValueError, match="Unicode"):
        PackageMetadata(
            id="org.comfyng.test",
            name=chr(0xD800),
            version="1.0.0",
            publisher="ComfyUI-NG",
            license="GPL-3.0-or-later",
        )


def test_runtime_policy_literals_match_the_normative_policy_list() -> None:
    assert LoadPolicy.LOAD_ON_EXECUTION.value == "LOAD_ON_EXECUTION"
    assert LoadPolicy.PRELOAD_ON_WORKFLOW_OPEN.value == "PRELOAD_ON_WORKFLOW_OPEN"
    assert LoadPolicy.PRELOAD_ON_QUEUE.value == "PRELOAD_ON_QUEUE"
    assert LoadPolicy.KEEP_WARM.value == "KEEP_WARM"
    assert UnloadPolicy.UNLOAD_AFTER_EXECUTION.value == "UNLOAD_AFTER_EXECUTION"
    assert UnloadPolicy.UNLOAD_AFTER_IDLE.value == "UNLOAD_AFTER_IDLE"
    assert UnloadPolicy.UNLOAD_ON_MEMORY_PRESSURE.value == "UNLOAD_ON_MEMORY_PRESSURE"
    assert UnloadPolicy.PERSISTENT.value == "PERSISTENT"


@pytest.mark.parametrize("version", ("1", "v1.0.0", "01.0.0", "1.0", "1.0.0.0"))
def test_manifest_rejects_non_semver_package_versions(
    tmp_path: Path,
    version: str,
) -> None:
    manifest_path = _write_manifest(tmp_path, version=version)

    with pytest.raises(ManifestValidationError, match="semantic version"):
        PluginManifest.load(manifest_path, root=tmp_path)


@pytest.mark.parametrize(
    "changes",
    (
        {"package_id": "ng-test"},
        {"node_id": "ng-test"},
    ),
    ids=("package", "node"),
)
def test_manifest_identifiers_require_a_real_dot_separator(
    tmp_path: Path,
    changes: dict[str, str],
) -> None:
    manifest_path = _write_manifest(tmp_path, **changes)

    with pytest.raises(ManifestValidationError, match="dotted"):
        PluginManifest.load(manifest_path, root=tmp_path)


def test_manifest_rejects_unsupported_schema_versions(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, schema_version=2)

    with pytest.raises(ManifestValidationError, match="schema_version"):
        PluginManifest.load(manifest_path, root=tmp_path)


@pytest.mark.parametrize(
    "schema",
    (
        [],
        {"type": "object"},
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "array",
        },
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {},
            "required": ["missing"],
        },
    ),
)
def test_manifest_rejects_invalid_json_schemas(tmp_path: Path, schema: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_schema(tmp_path / "schemas/input.json", schema)

    with pytest.raises(ManifestValidationError, match="schema"):
        PluginManifest.load(manifest_path, root=tmp_path)


def test_manifest_rejects_invalid_nested_json_schema_keywords(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_schema(
        tmp_path / "schemas/input.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"prompt": {"type": "not-a-json-schema-type"}},
        },
    )

    with pytest.raises(ManifestValidationError, match="type"):
        PluginManifest.load(manifest_path, root=tmp_path)


def test_manifest_validates_the_full_draft_2020_12_metaschema(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_schema(
        tmp_path / "schemas/input.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"guidance": {"type": "number", "minimum": "not-a-number"}},
        },
    )

    with pytest.raises(ManifestValidationError, match="Draft 2020-12"):
        PluginManifest.load(manifest_path, root=tmp_path)


def test_manifest_accepts_declared_optional_schema_properties(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_schema(
        tmp_path / "schemas/input.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "seed": {"type": "integer"},
            },
            "required": ["prompt"],
            "x-comfyng-optional": ["seed"],
        },
    )

    manifest = PluginManifest.load(manifest_path, root=tmp_path)

    assert manifest.nodes[0].input_schema["x-comfyng-optional"] == ("seed",)


@pytest.mark.parametrize(
    "optional",
    (
        "seed",
        [1],
        ["seed", "seed"],
        ["missing"],
        ["prompt"],
        [chr(0xD800)],
    ),
    ids=("not-array", "not-string", "duplicate", "unknown", "required", "unicode"),
)
def test_manifest_rejects_invalid_optional_schema_properties(
    tmp_path: Path,
    optional: object,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    properties = {
        "prompt": {"type": "string"},
        "seed": {"type": "integer"},
    }
    if optional == [chr(0xD800)]:
        properties[chr(0xD800)] = {"type": "string"}
    _write_schema(
        tmp_path / "schemas/input.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": properties,
            "required": ["prompt"],
            "x-comfyng-optional": optional,
        },
    )

    with pytest.raises(ManifestValidationError, match="x-comfyng-optional"):
        PluginManifest.load(manifest_path, root=tmp_path)


def test_manifest_rejects_schema_paths_outside_the_catalogue_root(
    tmp_path: Path,
) -> None:
    plugin_root = tmp_path / "plugin"
    outside = tmp_path / "outside.json"
    _write_schema(outside)
    manifest_path = _write_manifest(
        plugin_root,
        input_schema="../outside.json",
    )

    with pytest.raises(PathContainmentError):
        PluginManifest.load(manifest_path, root=plugin_root)


def test_manifest_rejects_unknown_fields_and_malformed_entrypoints(
    tmp_path: Path,
) -> None:
    malformed_entrypoint = _write_manifest(
        tmp_path / "one", entrypoint="not-an-entrypoint"
    )
    unknown_field = _write_manifest(tmp_path / "two", extra='unknown = "value"\n')

    with pytest.raises(ManifestValidationError, match="entrypoint"):
        PluginManifest.load(malformed_entrypoint, root=tmp_path / "one")
    with pytest.raises(ManifestValidationError, match="unknown"):
        PluginManifest.load(unknown_field, root=tmp_path / "two")
