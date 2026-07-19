from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from comfyng.plugins.lifecycle import PluginRuntimeManager
from comfyng.plugins.manifest import PluginManifest
from comfyng.plugins.permissions import PermissionSet
from comfyng.plugins.worker import SupervisorPluginBackend
from comfyng.workers.supervisor import RemoteExecutionError, WorkerSupervisor


SCHEMA = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": true,
  "properties": {}
}
"""


def _write_probe_plugin(root: Path) -> PluginManifest:
    source = root / "installed" / "org.comfyng.probe" / "1.0.0"
    (source / "schemas").mkdir(parents=True)
    (source / "package" / "probe_plugin").mkdir(parents=True)
    (source / "schemas" / "input.json").write_text(SCHEMA, encoding="utf-8")
    (source / "schemas" / "output.json").write_text(SCHEMA, encoding="utf-8")
    (source / "package" / "probe_plugin" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (source / "package" / "probe_plugin" / "runtime.py").write_text(
        """
from pathlib import Path
import socket
import subprocess

class Runtime:
    def execute(self, operation, payload, cancellation):
        if operation == "read":
            return {"text": Path(payload["path"]).read_text(encoding="utf-8")}
        if operation == "write":
            Path(payload["path"]).write_text(payload["text"], encoding="utf-8")
            return {"written": True}
        if operation == "network":
            socket.create_connection(("127.0.0.1", 9), timeout=0.1)
        if operation == "subprocess":
            subprocess.run(["true"], check=True)
        raise ValueError(operation)

def create_runtime():
    return Runtime()
""".lstrip(),
        encoding="utf-8",
    )
    (source / "ng-node.toml").write_text(
        """
schema_version = 1
[package]
id = "org.comfyng.probe"
name = "Permission Probe"
version = "1.0.0"
publisher = "Tests"
license = "GPL-3.0-or-later"
[runtime]
language = "python"
python = ">=3.14"
entrypoint = "probe_plugin.runtime:create_runtime"
isolation = "plugin_worker"
load_policy = "LOAD_ON_EXECUTION"
unload_policy = "UNLOAD_AFTER_IDLE"
idle_timeout_seconds = 60
[resources]
gpu = "none"
estimated_ram_mb = 16
estimated_vram_mb = 0
network = false
[permissions]
network = false
filesystem_read = ["input"]
filesystem_write = ["output"]
subprocess = false
gpu = false
camera = false
microphone = false
[[nodes]]
id = "ng.probe.permissions"
display_name = "Permission Probe"
input_schema = "schemas/input.json"
output_schema = "schemas/output.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return PluginManifest.load(source / "ng-node.toml", root=source)


def test_third_party_runtime_cannot_exceed_declared_permissions(tmp_path: Path) -> None:
    async def scenario() -> None:
        plugin_root = tmp_path / "plugins"
        readable = tmp_path / "input"
        writable = tmp_path / "output"
        forbidden = tmp_path / "private"
        for directory in (plugin_root, readable, writable, forbidden):
            directory.mkdir()
        (readable / "allowed.txt").write_text("allowed", encoding="utf-8")
        (forbidden / "secret.txt").write_text("secret", encoding="utf-8")
        manifest = _write_probe_plugin(plugin_root)
        permissions = PermissionSet(
            filesystem_read=("input",),
            filesystem_write=("output",),
        )

        with WorkerSupervisor() as supervisor:
            backend = SupervisorPluginBackend(
                supervisor,
                plugin_root=plugin_root,
                permission_roots={"input": readable, "output": writable},
            )
            manager = PluginRuntimeManager(backend)
            manager.register(manifest, manifest.source_path.parent, permissions)

            assert await manager.execute(
                manifest.package.id,
                "read",
                {"path": str(readable / "allowed.txt")},
            ) == {"text": "allowed"}
            await manager.execute(
                manifest.package.id,
                "write",
                {"path": str(writable / "created.txt"), "text": "created"},
            )

            for operation, payload, message in (
                (
                    "read",
                    {"path": str(forbidden / "secret.txt")},
                    "filesystem read denied",
                ),
                (
                    "write",
                    {"path": str(forbidden / "stolen.txt"), "text": "stolen"},
                    "filesystem write denied",
                ),
                ("network", {}, "network access denied"),
                ("subprocess", {}, "subprocess creation denied"),
            ):
                with pytest.raises(RemoteExecutionError, match=message):
                    await manager.execute(
                        manifest.package.id,
                        operation,
                        payload,
                    )

            assert await manager.unload(manifest.package.id) is True

        assert (writable / "created.txt").read_text(encoding="utf-8") == "created"
        assert not (forbidden / "stolen.txt").exists()

    asyncio.run(scenario())
