from __future__ import annotations

from pathlib import Path

import pytest

from comfyng.plugins.environments import EnvironmentManager
from comfyng.plugins.installer import (
    InMemoryPluginRegistry,
    InstallPhase,
    PluginInstallError,
    PluginInstaller,
    bundle_digest,
)
from comfyng.plugins.signatures import HMACSignatureVerifier


SCHEMA = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "properties": {"value": {"type": "integer"}},
  "required": ["value"]
}
"""


def write_plugin(root: Path) -> Path:
    source = root / "source"
    (source / "schemas").mkdir(parents=True)
    (source / "package" / "demo_plugin").mkdir(parents=True)
    (source / "schemas" / "input.json").write_text(SCHEMA, encoding="utf-8")
    (source / "schemas" / "output.json").write_text(SCHEMA, encoding="utf-8")
    (source / "package" / "demo_plugin" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (source / "package" / "demo_plugin" / "runtime.py").write_text(
        "def create_runtime():\n    return {'runtime': 'demo'}\n",
        encoding="utf-8",
    )
    (source / "ng-node.toml").write_text(
        """
schema_version = 1
dependencies = []

[package]
id = "org.comfyng.demo"
name = "Demo Plugin"
version = "1.2.3"
publisher = "Tests"
license = "GPL-3.0-or-later"

[runtime]
language = "python"
python = ">=3.14"
entrypoint = "demo_plugin.runtime:create_runtime"
isolation = "plugin_worker"
load_policy = "LOAD_ON_EXECUTION"
unload_policy = "UNLOAD_AFTER_IDLE"
idle_timeout_seconds = 30

[resources]
gpu = "none"
estimated_ram_mb = 16
estimated_vram_mb = 0
network = false

[permissions]
network = false
filesystem_read = ["models", "input"]
filesystem_write = ["output", "temp"]
subprocess = false
gpu = false
camera = false
microphone = false

[[nodes]]
id = "ng.demo.identity"
display_name = "Demo Identity"
input_schema = "schemas/input.json"
output_schema = "schemas/output.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return source


class FastEnvironmentManager(EnvironmentManager):
    def create(self, bundle: Path, lockfile: Path) -> Path:
        environment = bundle / ".venv"
        environment.mkdir()
        (environment / "created").write_text(lockfile.read_text(encoding="utf-8"))
        return environment

    def test_import(self, environment: Path, bundle: Path, entrypoint: str) -> None:
        if (
            not environment.is_dir()
            or entrypoint != "demo_plugin.runtime:create_runtime"
        ):
            raise RuntimeError("invalid import fixture")


def configured_installer(
    root: Path,
    *,
    observer=None,
) -> tuple[PluginInstaller, InMemoryPluginRegistry, HMACSignatureVerifier]:
    registry = InMemoryPluginRegistry()
    verifier = HMACSignatureVerifier({"Tests": b"unit-test-secret"})
    installer = PluginInstaller(
        root,
        registry=registry,
        signature_verifier=verifier,
        environment_manager=FastEnvironmentManager(),
        phase_observer=observer,
    )
    return installer, registry, verifier


@pytest.mark.parametrize("failing_phase", tuple(InstallPhase))
def test_every_install_phase_rolls_back_atomically(
    tmp_path: Path,
    failing_phase: InstallPhase,
) -> None:
    source = write_plugin(tmp_path)
    digest = bundle_digest(source)

    def observer(phase: InstallPhase) -> None:
        if phase is failing_phase:
            raise RuntimeError(f"injected failure at {phase.value}")

    installer, registry, verifier = configured_installer(
        tmp_path / "plugins",
        observer=observer,
    )

    with pytest.raises(PluginInstallError) as captured:
        installer.install(
            source,
            expected_sha256=digest,
            signature=verifier.sign("Tests", digest),
        )

    assert captured.value.phase is failing_phase
    assert registry.records == {}
    assert not tuple(installer.staging_root.iterdir())
    assert not tuple(installer.installed_root.rglob("ng-node.toml"))


def test_successful_install_publishes_complete_layout_and_registry_record(
    tmp_path: Path,
) -> None:
    source = write_plugin(tmp_path)
    digest = bundle_digest(source)
    installer, registry, verifier = configured_installer(tmp_path / "plugins")

    result = installer.install(
        source,
        expected_sha256=digest,
        signature=verifier.sign("Tests", digest),
    )

    assert result.package_id == "org.comfyng.demo"
    assert result.version == "1.2.3"
    assert result.digest == digest
    assert (
        result.path
        == (installer.installed_root / "org.comfyng.demo" / "1.2.3").resolve()
    )
    assert (result.path / "ng-node.toml").is_file()
    assert (result.path / "lockfile").is_file()
    assert (result.path / "package" / "demo_plugin" / "runtime.py").is_file()
    assert (result.path / "schemas" / "input.json").is_file()
    assert (result.path / ".venv").is_dir()
    assert registry.records[(result.package_id, result.version)] == result
    assert result.permissions.filesystem_read == ("input", "models")
    assert not tuple(installer.staging_root.iterdir())


def test_hash_and_signature_mismatch_fail_before_publication(tmp_path: Path) -> None:
    source = write_plugin(tmp_path)
    digest = bundle_digest(source)
    installer, registry, verifier = configured_installer(tmp_path / "plugins")

    with pytest.raises(PluginInstallError) as bad_signature:
        installer.install(
            source,
            expected_sha256=digest,
            signature="0" * 64,
        )
    assert bad_signature.value.phase is InstallPhase.VERIFY_SIGNATURE

    with pytest.raises(PluginInstallError) as bad_hash:
        installer.install(
            source,
            expected_sha256="f" * 64,
            signature=verifier.sign("Tests", "f" * 64),
        )
    assert bad_hash.value.phase is InstallPhase.VERIFY_HASH
    assert registry.records == {}


def test_unreadable_manifest_is_reported_in_signature_phase(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "ng-node.toml").write_text("[package\n", encoding="utf-8")
    digest = bundle_digest(source)
    installer, registry, _verifier = configured_installer(tmp_path / "plugins")

    with pytest.raises(PluginInstallError) as captured:
        installer.install(
            source,
            expected_sha256=digest,
            signature="0" * 64,
        )

    assert captured.value.phase is InstallPhase.VERIFY_SIGNATURE
    assert registry.records == {}
    assert not tuple(installer.staging_root.iterdir())


def test_bundle_digest_is_content_and_path_derived(tmp_path: Path) -> None:
    source = write_plugin(tmp_path)
    first = bundle_digest(source)
    copied = tmp_path / "copy"
    import shutil

    shutil.copytree(source, copied)

    assert bundle_digest(copied) == first
    (copied / "package" / "demo_plugin" / "runtime.py").write_text(
        "def create_runtime():\n    return {'changed': True}\n",
        encoding="utf-8",
    )
    assert bundle_digest(copied) != first
