from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

import pytest

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


def plugin_source(root: Path) -> Path:
    source = root / "source"
    (source / "schemas").mkdir(parents=True)
    (source / "package" / "atomic_plugin").mkdir(parents=True)
    for name in ("input", "output"):
        (source / "schemas" / f"{name}.json").write_text(SCHEMA, encoding="utf-8")
    (source / "package" / "atomic_plugin" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (source / "package" / "atomic_plugin" / "runtime.py").write_text(
        """
class Runtime:
    def execute(self, operation, payload, cancellation):
        return {"operation": operation, "payload": dict(payload)}
    def unload(self):
        return {"released": 1}

def create_runtime():
    return Runtime()
""".lstrip(),
        encoding="utf-8",
    )
    (source / "ng-node.toml").write_text(
        """
schema_version = 1
dependencies = []
[package]
id = "org.comfyng.atomic"
name = "Atomic Plugin"
version = "1.0.0"
publisher = "Tests"
license = "GPL-3.0-or-later"
[runtime]
language = "python"
python = ">=3.14"
entrypoint = "atomic_plugin.runtime:create_runtime"
isolation = "plugin_worker"
load_policy = "LOAD_ON_EXECUTION"
unload_policy = "UNLOAD_AFTER_EXECUTION"
idle_timeout_seconds = 0
[resources]
gpu = "none"
estimated_ram_mb = 8
estimated_vram_mb = 0
network = false
[permissions]
network = false
filesystem_read = []
filesystem_write = []
subprocess = false
gpu = false
camera = false
microphone = false
[[nodes]]
id = "ng.atomic.echo"
display_name = "Atomic Echo"
input_schema = "schemas/input.json"
output_schema = "schemas/output.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return source


def test_real_environment_install_is_atomic_and_importable(tmp_path: Path) -> None:
    source = plugin_source(tmp_path)
    digest = bundle_digest(source)
    verifier = HMACSignatureVerifier({"Tests": b"integration-secret"})
    registry = InMemoryPluginRegistry()
    installer = PluginInstaller(
        tmp_path / "plugins",
        registry=registry,
        signature_verifier=verifier,
    )

    result = installer.install(
        source,
        expected_sha256=digest,
        signature=verifier.sign("Tests", digest),
    )

    python = result.path / ".venv" / "bin" / "python"
    assert python.is_file()
    assert result.path.is_dir()
    assert registry.records[(result.package_id, result.version)] == result
    assert not tuple(installer.staging_root.iterdir())


def test_concurrent_publish_never_exposes_a_partial_second_install(
    tmp_path: Path,
) -> None:
    source = plugin_source(tmp_path)
    digest = bundle_digest(source)
    verifier = HMACSignatureVerifier({"Tests": b"integration-secret"})
    registry = InMemoryPluginRegistry()
    installer = PluginInstaller(
        tmp_path / "plugins",
        registry=registry,
        signature_verifier=verifier,
    )
    signature = verifier.sign("Tests", digest)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(
                installer.install,
                source,
                expected_sha256=digest,
                signature=signature,
            )
            for _ in range(2)
        )
    results = []
    failures = []
    for future in futures:
        try:
            results.append(future.result())
        except PluginInstallError as exc:
            failures.append(exc)

    assert len(results) == 1
    assert len(failures) == 1
    assert failures[0].phase is InstallPhase.PUBLISH
    installed = results[0].path
    assert (installed / "ng-node.toml").is_file()
    assert (installed / ".venv" / "bin" / "python").is_file()
    assert len(registry.records) == 1
    assert not tuple(installer.staging_root.iterdir())


def test_zip_path_traversal_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "owned")
    verifier = HMACSignatureVerifier({"Tests": b"integration-secret"})
    installer = PluginInstaller(
        tmp_path / "plugins",
        registry=InMemoryPluginRegistry(),
        signature_verifier=verifier,
    )

    with pytest.raises(PluginInstallError) as captured:
        installer.install(
            archive,
            expected_sha256="0" * 64,
            signature="0" * 64,
        )

    assert captured.value.phase is InstallPhase.DOWNLOAD
    assert not (tmp_path / "escape.txt").exists()
