from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

from comfyng.core.enums import (
    GpuRequirement,
    LifecycleState,
    LoadPolicy,
    RuntimeIsolation,
    UnloadPolicy,
)
from comfyng.plugins.lifecycle import (
    PluginBusyError,
    PluginRuntimeManager,
    PluginWorkerBackend,
    RuntimePluginSpec,
)
from comfyng.plugins.manifest import (
    NodeDefinition,
    PackageMetadata,
    PluginManifest,
    ResourceRequirements,
    RuntimeDefinition,
)
from comfyng.plugins.permissions import PermissionSet


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
}


def manifest(
    package_id: str,
    *,
    load_policy: LoadPolicy = LoadPolicy.LOAD_ON_EXECUTION,
    unload_policy: UnloadPolicy = UnloadPolicy.UNLOAD_AFTER_IDLE,
    idle_timeout: int = 30,
) -> PluginManifest:
    suffix = package_id.rsplit(".", 1)[-1].replace("-", "_")
    return PluginManifest(
        schema_version=1,
        package=PackageMetadata(
            id=package_id,
            name=package_id,
            version="1.0.0",
            publisher="Tests",
            license="GPL-3.0-or-later",
        ),
        runtime=RuntimeDefinition(
            language="python",
            python=">=3.14",
            entrypoint=f"sentinel_{suffix}:create_runtime",
            isolation=RuntimeIsolation.PLUGIN_WORKER,
            load_policy=load_policy,
            unload_policy=unload_policy,
            idle_timeout_seconds=idle_timeout,
        ),
        resources=ResourceRequirements(
            gpu=GpuRequirement.NONE,
            estimated_ram_mb=16,
            estimated_vram_mb=0,
            network=False,
        ),
        nodes=(
            NodeDefinition(
                id=f"ng.test.{suffix}",
                version="1.0.0",
                display_name=suffix,
                package_id=package_id,
                input_schema_path=Path(f"/tmp/{suffix}.input.json"),
                output_schema_path=Path(f"/tmp/{suffix}.output.json"),
                input_schema=SCHEMA,
                output_schema=SCHEMA,
            ),
        ),
        source_path=Path(f"/tmp/{suffix}/ng-node.toml"),
    )


class FakeBackend(PluginWorkerBackend):
    def __init__(self) -> None:
        self.started: list[tuple[str, PermissionSet]] = []
        self.loaded: list[tuple[str, RuntimePluginSpec]] = []
        self.executed: list[tuple[str, str, str, object]] = []
        self.unloaded: list[tuple[str, str]] = []
        self.stopped: list[str] = []
        self.fail_packages: set[str] = set()

    async def start_group(self, group_id: str, permissions: PermissionSet) -> str:
        self.started.append((group_id, permissions))
        return f"worker:{group_id}"

    async def load_plugin(self, worker_id: str, spec: RuntimePluginSpec) -> None:
        self.loaded.append((worker_id, spec))

    async def execute(
        self,
        worker_id: str,
        package_id: str,
        node_id: str,
        payload: object,
    ) -> object:
        self.executed.append((worker_id, package_id, node_id, payload))
        if package_id in self.fail_packages:
            raise RuntimeError("plugin crashed")
        return {"worker": worker_id, "node": node_id, "payload": payload}

    async def unload_plugin(self, worker_id: str, package_id: str) -> None:
        self.unloaded.append((worker_id, package_id))

    async def stop_group(self, worker_id: str) -> None:
        self.stopped.append(worker_id)


def test_jit_load_execute_and_real_worker_unload_without_core_import() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        plugin = manifest("org.comfyng.alpha")
        module_name = plugin.runtime.entrypoint.partition(":")[0]
        sys.modules.pop(module_name, None)
        manager = PluginRuntimeManager(backend, clock=lambda: 100.0)
        manager.register(
            plugin,
            Path("/tmp/plugins/alpha").resolve(),
            PermissionSet(),
        )

        assert manager.state(plugin.package.id) is LifecycleState.DISCOVERED
        assert module_name not in sys.modules
        result = await manager.execute(
            plugin.package.id,
            plugin.nodes[0].id,
            {"value": 4},
        )

        assert result["payload"] == {"value": 4}
        assert manager.state(plugin.package.id) is LifecycleState.IDLE
        assert manager.history(plugin.package.id) == (
            LifecycleState.DISCOVERED,
            LifecycleState.RESOLVED,
            LifecycleState.PRELOADING,
            LifecycleState.LOADED,
            LifecycleState.READY,
            LifecycleState.BUSY,
            LifecycleState.IDLE,
        )
        assert module_name not in sys.modules

        assert await manager.unload(plugin.package.id) is True
        assert manager.state(plugin.package.id) is LifecycleState.UNLOADED
        assert backend.unloaded == [
            ("worker:plugin:org.comfyng.alpha", plugin.package.id)
        ]
        assert backend.stopped == ["worker:plugin:org.comfyng.alpha"]

    asyncio.run(scenario())


def test_trust_group_shares_one_worker_and_stops_after_last_plugin() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        manager = PluginRuntimeManager(backend)
        alpha = manifest("org.comfyng.alpha")
        beta = manifest("org.comfyng.beta")
        for plugin in (alpha, beta):
            manager.register(
                plugin,
                Path(f"/tmp/{plugin.package.id}").resolve(),
                PermissionSet(network=False),
                trust_group="official",
            )
            await manager.load(plugin.package.id)

        assert len(backend.started) == 1
        assert backend.started[0][0] == "trust:official"
        await manager.unload(alpha.package.id)
        assert backend.stopped == []
        await manager.unload(beta.package.id)
        assert backend.stopped == ["worker:trust:official"]

    asyncio.run(scenario())


def test_trust_group_serializes_execution_on_its_single_worker() -> None:
    class SerialProbeBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum_active = 0

        async def execute(
            self,
            worker_id: str,
            package_id: str,
            node_id: str,
            payload: object,
        ) -> object:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return await super().execute(
                    worker_id,
                    package_id,
                    node_id,
                    payload,
                )
            finally:
                self.active -= 1

    async def scenario() -> None:
        backend = SerialProbeBackend()
        manager = PluginRuntimeManager(backend)
        alpha = manifest("org.comfyng.alpha")
        beta = manifest("org.comfyng.beta")
        for plugin in (alpha, beta):
            manager.register(
                plugin,
                Path(f"/tmp/{plugin.package.id}").resolve(),
                PermissionSet(),
                trust_group="official",
            )

        await asyncio.gather(
            manager.execute(alpha.package.id, alpha.nodes[0].id, {}),
            manager.execute(beta.package.id, beta.nodes[0].id, {}),
        )

        assert backend.maximum_active == 1

    asyncio.run(scenario())


def test_crashing_plugin_only_fails_its_isolation_group() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        manager = PluginRuntimeManager(backend)
        alpha = manifest("org.comfyng.alpha")
        beta = manifest("org.comfyng.beta")
        for plugin in (alpha, beta):
            manager.register(
                plugin,
                Path(f"/tmp/{plugin.package.id}").resolve(),
                PermissionSet(),
            )
        backend.fail_packages.add(alpha.package.id)

        with pytest.raises(RuntimeError, match="plugin crashed"):
            await manager.execute(alpha.package.id, alpha.nodes[0].id, {})

        assert manager.state(alpha.package.id) is LifecycleState.FAILED
        assert manager.state(beta.package.id) is LifecycleState.DISCOVERED
        result = await manager.execute(beta.package.id, beta.nodes[0].id, {})
        assert result["node"] == beta.nodes[0].id
        assert manager.state(beta.package.id) is LifecycleState.IDLE

    asyncio.run(scenario())


def test_idle_and_memory_pressure_policies_unload_only_eligible_plugins() -> None:
    async def scenario() -> None:
        now = [0.0]
        backend = FakeBackend()
        manager = PluginRuntimeManager(backend, clock=lambda: now[0])
        idle = manifest(
            "org.comfyng.idle",
            unload_policy=UnloadPolicy.UNLOAD_AFTER_IDLE,
            idle_timeout=5,
        )
        pressure = manifest(
            "org.comfyng.pressure",
            unload_policy=UnloadPolicy.UNLOAD_ON_MEMORY_PRESSURE,
        )
        persistent = manifest(
            "org.comfyng.persistent",
            unload_policy=UnloadPolicy.PERSISTENT,
        )
        for plugin in (idle, pressure, persistent):
            manager.register(
                plugin,
                Path(f"/tmp/{plugin.package.id}").resolve(),
                PermissionSet(),
            )
            await manager.execute(plugin.package.id, plugin.nodes[0].id, {})

        now[0] = 4.9
        assert await manager.sweep_idle() == ()
        now[0] = 5.0
        assert await manager.sweep_idle() == (idle.package.id,)
        assert await manager.handle_memory_pressure() == (pressure.package.id,)
        assert manager.state(persistent.package.id) is LifecycleState.IDLE

    asyncio.run(scenario())


def test_preload_policies_and_busy_unload_guard() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        manager = PluginRuntimeManager(backend)
        workflow = manifest(
            "org.comfyng.workflow",
            load_policy=LoadPolicy.PRELOAD_ON_WORKFLOW_OPEN,
        )
        queued = manifest(
            "org.comfyng.queued",
            load_policy=LoadPolicy.PRELOAD_ON_QUEUE,
        )
        keep_warm = manifest(
            "org.comfyng.keep-warm",
            load_policy=LoadPolicy.KEEP_WARM,
            unload_policy=UnloadPolicy.PERSISTENT,
        )
        for plugin in (workflow, queued, keep_warm):
            manager.register(
                plugin,
                Path(f"/tmp/{plugin.package.id}").resolve(),
                PermissionSet(),
            )

        assert await manager.preload_for_workflow(
            (workflow.package.id, queued.package.id)
        ) == (workflow.package.id,)
        assert await manager.preload_for_queue(
            (workflow.package.id, queued.package.id)
        ) == (queued.package.id,)
        assert await manager.preload_keep_warm() == (keep_warm.package.id,)
        assert manager.state(keep_warm.package.id) is LifecycleState.READY

        record = manager.record(workflow.package.id)
        record.transition(LifecycleState.BUSY)
        with pytest.raises(PluginBusyError):
            await manager.unload(workflow.package.id)

    asyncio.run(scenario())
