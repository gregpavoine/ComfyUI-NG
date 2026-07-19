from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from comfyng.core.enums import LifecycleState
from comfyng.plugins.lifecycle import PluginRuntimeManager
from comfyng.plugins.manifest import PluginManifest
from comfyng.plugins.permissions import PermissionSet
from comfyng.plugins.worker import SupervisorPluginBackend
from comfyng.workers.supervisor import WorkerSupervisor


SCHEMA = """{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "properties": {"value": {"type": "integer"}},
  "required": ["value"]
}
"""


def write_runtime_plugin(root: Path) -> PluginManifest:
    source = root / "installed" / "org.comfyng.jit" / "1.0.0"
    (source / "schemas").mkdir(parents=True)
    (source / "package" / "jit_plugin").mkdir(parents=True)
    (source / "schemas" / "input.json").write_text(SCHEMA, encoding="utf-8")
    (source / "schemas" / "output.json").write_text(SCHEMA, encoding="utf-8")
    (source / "package" / "jit_plugin" / "__init__.py").write_text("", encoding="utf-8")
    (source / "package" / "jit_plugin" / "runtime.py").write_text(
        """
class Runtime:
    def __init__(self):
        self.values = []
    def execute(self, operation, payload, cancellation):
        self.values.append(payload["value"])
        return {"operation": operation, "sum": sum(self.values)}
    def unload(self):
        released = len(self.values)
        self.values.clear()
        return {"released": released}

def create_runtime():
    return Runtime()
""".lstrip(),
        encoding="utf-8",
    )
    (source / "ng-node.toml").write_text(
        """
schema_version = 1
[package]
id = "org.comfyng.jit"
name = "JIT Plugin"
version = "1.0.0"
publisher = "Tests"
license = "GPL-3.0-or-later"
[runtime]
language = "python"
python = ">=3.14"
entrypoint = "jit_plugin.runtime:create_runtime"
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
filesystem_read = []
filesystem_write = []
subprocess = false
gpu = false
camera = false
microphone = false
[[nodes]]
id = "ng.jit.accumulate"
display_name = "JIT Accumulate"
input_schema = "schemas/input.json"
output_schema = "schemas/output.json"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return PluginManifest.load(source / "ng-node.toml", root=source)


def test_runtime_code_is_jit_loaded_only_in_worker_and_unloaded_by_worker_stop(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = write_runtime_plugin(tmp_path)
        install_path = manifest.source_path.parent
        module_name = manifest.runtime.entrypoint.partition(":")[0]
        sys.modules.pop(module_name, None)
        with WorkerSupervisor() as supervisor:
            backend = SupervisorPluginBackend(
                supervisor,
                plugin_root=tmp_path,
            )
            manager = PluginRuntimeManager(backend)
            manager.register(manifest, install_path, PermissionSet())

            first = await manager.execute(
                manifest.package.id,
                manifest.nodes[0].id,
                {"value": 2},
            )
            second = await manager.execute(
                manifest.package.id,
                manifest.nodes[0].id,
                {"value": 3},
            )
            record = manager.record(manifest.package.id)
            assert record.worker_id is not None
            worker_id = record.worker_id
            pid = supervisor.snapshot(worker_id).pid

            assert first["sum"] == 2
            assert second["sum"] == 5
            assert module_name not in sys.modules
            assert await manager.unload(manifest.package.id) is True
            assert manager.state(manifest.package.id) is LifecycleState.UNLOADED
            assert supervisor.snapshot(worker_id).status == "stopped"
            assert pid > 0
            assert supervisor.snapshot(worker_id).pid == 0

    asyncio.run(scenario())
