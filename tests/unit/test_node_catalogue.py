from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from comfyng.core.errors import DuplicateNodeDefinitionError
from comfyng.plugins.catalogue import NodeCatalogue


OFFICIAL_DISPLAY_NAMES = {
    "NG Model Loader",
    "NG Text Encoder Loader",
    "NG VAE Loader",
    "NG Model Inspector",
    "NG Model Unload",
    "NG LoRA Loader",
    "NG LoRA Stack",
    "NG LoRA Inspector",
    "NG Prompt Encode",
    "NG Guidance",
    "NG Conditioning Combine",
    "NG Conditioning Mask",
    "NG Empty Latent",
    "NG Image To Latent",
    "NG Latent To Image",
    "NG Latent Resize",
    "NG Latent Blend",
    "NG Sampler",
    "NG Sampler Advanced",
    "NG Noise",
    "NG Scheduler",
    "NG Load Image",
    "NG Save Image",
    "NG Preview Image",
    "NG Resize Image",
    "NG Crop Image",
    "NG Image Metadata",
    "NG Switch",
    "NG Compare",
    "NG Route",
    "NG Merge",
    "NG For Each",
    "NG Collect",
    "NG Subgraph Input",
    "NG Subgraph Output",
    "NG Job Info",
    "NG Hardware Info",
    "NG Memory Policy",
    "NG Performance Profile",
    "NG Cache Control",
}
EMPTY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}


def _write_plugin(
    root: Path,
    *,
    package_id: str,
    node_id: str,
    display_name: str = "Sentinel Node",
) -> None:
    schemas = root / "schemas"
    schemas.mkdir(parents=True)
    for name in ("input.json", "output.json"):
        (schemas / name).write_text(json.dumps(EMPTY_SCHEMA), encoding="utf-8")
    (root / "ng-node.toml").write_text(
        f'''schema_version = 1

[package]
id = "{package_id}"
name = "Sentinel Runtime"
version = "1.0.0"
publisher = "Tests"
license = "GPL-3.0-or-later"

[runtime]
language = "python"
python = ">=3.14"
entrypoint = "sentinel_runtime:create_runtime"
isolation = "plugin_worker"
load_policy = "LOAD_ON_EXECUTION"
unload_policy = "UNLOAD_AFTER_EXECUTION"
idle_timeout_seconds = 0

[resources]
gpu = "none"
estimated_ram_mb = 8
estimated_vram_mb = 0
network = false

[[nodes]]
id = "{node_id}"
display_name = "{display_name}"
input_schema = "schemas/input.json"
output_schema = "schemas/output.json"
''',
        encoding="utf-8",
    )


def test_official_catalogue_exposes_exactly_40_display_names() -> None:
    catalogue = NodeCatalogue.discover()

    assert len(catalogue.nodes) == 40
    assert set(catalogue.display_names) == OFFICIAL_DISPLAY_NAMES
    assert catalogue.get("ng.sample.run", "1.0.0").display_name == "NG Sampler"
    assert all(node.input_schema["type"] == "object" for node in catalogue.nodes)
    assert all(node.output_schema["type"] == "object" for node in catalogue.nodes)


def test_official_schemas_declare_every_required_or_optional_port() -> None:
    catalogue = NodeCatalogue.discover()

    for node in catalogue.nodes:
        input_properties = set(node.input_schema["properties"])
        input_required = set(node.input_schema.get("required", ()))
        input_optional = set(node.input_schema.get("x-comfyng-optional", ()))
        assert node.input_schema["additionalProperties"] is False, node.id
        assert input_required.isdisjoint(input_optional), node.id
        assert input_required | input_optional == input_properties, node.id

        output_properties = node.output_schema["properties"]
        assert output_properties, node.id
        assert node.output_schema["additionalProperties"] is False, node.id
        assert set(node.output_schema.get("required", ())) == set(output_properties), node.id
        assert all(definition for definition in output_properties.values()), node.id


def test_discovery_never_imports_a_runtime_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sentinel_runtime.py").write_text(
        'raise AssertionError("runtime entrypoint was imported during discovery")\n',
        encoding="utf-8",
    )
    _write_plugin(
        tmp_path / "runtime",
        package_id="org.comfyng.sentinel",
        node_id="ng.test.sentinel",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("sentinel_runtime", None)

    catalogue = NodeCatalogue.discover(tmp_path)

    assert catalogue.display_names == ("Sentinel Node",)
    assert "sentinel_runtime" not in sys.modules


def test_catalogue_rejects_duplicate_node_id_and_version(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "one",
        package_id="org.comfyng.one",
        node_id="ng.test.duplicate",
    )
    _write_plugin(
        tmp_path / "two",
        package_id="org.comfyng.two",
        node_id="ng.test.duplicate",
    )

    with pytest.raises(DuplicateNodeDefinitionError, match="ng.test.duplicate"):
        NodeCatalogue.discover(tmp_path)


def test_catalogue_rejects_duplicate_package_versions(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "one",
        package_id="org.comfyng.same",
        node_id="ng.test.one",
    )
    _write_plugin(
        tmp_path / "two",
        package_id="org.comfyng.same",
        node_id="ng.test.two",
    )

    with pytest.raises(DuplicateNodeDefinitionError, match="package"):
        NodeCatalogue.discover(tmp_path)
