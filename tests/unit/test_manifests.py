from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfyng.core.enums import LoadPolicy, UnloadPolicy
from comfyng.core.errors import ManifestValidationError, PathContainmentError
from comfyng.plugins.manifest import PluginManifest


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
    assert manifest.nodes[0].input_schema_path == (tmp_path / "schemas/input.json").resolve()
    assert PluginManifest.from_json(manifest.to_json()) == manifest
    with pytest.raises(AttributeError):
        manifest.schema_version = 2  # type: ignore[misc]


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
            "properties": {
                "guidance": {"type": "number", "minimum": "not-a-number"}
            },
        },
    )

    with pytest.raises(ManifestValidationError, match="Draft 2020-12"):
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
    malformed_entrypoint = _write_manifest(tmp_path / "one", entrypoint="not-an-entrypoint")
    unknown_field = _write_manifest(tmp_path / "two", extra='unknown = "value"\n')

    with pytest.raises(ManifestValidationError, match="entrypoint"):
        PluginManifest.load(malformed_entrypoint, root=tmp_path / "one")
    with pytest.raises(ManifestValidationError, match="unknown"):
        PluginManifest.load(unknown_field, root=tmp_path / "two")
