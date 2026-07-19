from __future__ import annotations

import asyncio
from collections.abc import Mapping
from hashlib import sha256
import importlib
import inspect
from pathlib import Path
import sys
import threading
from typing import Any

from comfyng.plugins.lifecycle import PluginWorkerBackend, RuntimePluginSpec
from comfyng.plugins.permissions import PermissionSet
from comfyng.workers.protocol import WorkerKind, WorkerSpec
from comfyng.workers.sandbox import SandboxPolicy
from comfyng.workers.supervisor import WorkerSupervisor


class PluginWorkerError(RuntimeError):
    pass


class _LoadedRuntime:
    def __init__(self, package_id: str, package_root: Path, runtime: object) -> None:
        self.package_id = package_id
        self.package_root = package_root
        self.runtime = runtime

    def execute(
        self,
        node_id: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        execute_node = getattr(self.runtime, "execute_node", None)
        if callable(execute_node):
            return execute_node(node_id, payload, cancellation)
        execute = getattr(self.runtime, "execute", None)
        if callable(execute):
            return execute(node_id, payload, cancellation)
        if callable(self.runtime):
            return self.runtime(node_id, payload, cancellation)
        raise PluginWorkerError(
            f"plugin {self.package_id!r} runtime has no execution interface"
        )

    def unload(self) -> object:
        unload = getattr(self.runtime, "unload", None)
        return unload() if callable(unload) else {"released": 0}


class PluginMultiplexerRuntime:
    """Worker-local JIT loader; third-party modules never enter the core process."""

    def __init__(self) -> None:
        self._runtimes: dict[str, _LoadedRuntime] = {}

    @staticmethod
    def _string(payload: Mapping[str, Any], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"plugin {name} must be a non-empty string")
        return value

    def _load(self, payload: Mapping[str, Any]) -> dict[str, object]:
        package_id = self._string(payload, "package_id")
        entrypoint = self._string(payload, "entrypoint")
        install_path = Path(self._string(payload, "install_path")).resolve(strict=True)
        if not install_path.is_dir():
            raise ValueError("plugin install_path must be a directory")
        package_root = (install_path / "package").resolve(strict=True)
        if not package_root.is_dir() or not package_root.is_relative_to(install_path):
            raise ValueError("plugin package directory is invalid")
        if package_id in self._runtimes:
            return {"loaded": package_id, "already_loaded": True}
        module_name, separator, attribute = entrypoint.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("plugin entrypoint is invalid")
        sys.path.insert(0, str(package_root))
        try:
            target = getattr(importlib.import_module(module_name), attribute)
        finally:
            try:
                sys.path.remove(str(package_root))
            except ValueError:
                pass
        runtime = target() if inspect.isclass(target) or callable(target) else target
        self._runtimes[package_id] = _LoadedRuntime(
            package_id,
            package_root,
            runtime,
        )
        return {"loaded": package_id, "already_loaded": False}

    def _unload_one(self, package_id: str) -> dict[str, object]:
        loaded = self._runtimes.pop(package_id, None)
        if loaded is None:
            return {"unloaded": package_id, "released": False}
        result = loaded.unload()
        for module_name, module in tuple(sys.modules.items()):
            source = getattr(module, "__file__", None)
            if not isinstance(source, str):
                continue
            try:
                module_path = Path(source).resolve()
            except OSError, RuntimeError:
                continue
            if module_path.is_relative_to(loaded.package_root):
                sys.modules.pop(module_name, None)
        return {"unloaded": package_id, "released": True, "result": result}

    def execute(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        if operation == "plugin.load":
            return self._load(payload)
        if operation == "plugin.unload":
            return self._unload_one(self._string(payload, "package_id"))
        if operation == "plugin.execute":
            package_id = self._string(payload, "package_id")
            node_id = self._string(payload, "node_id")
            node_payload = payload.get("payload", {})
            if not isinstance(node_payload, Mapping):
                raise ValueError("plugin payload must be a mapping")
            try:
                loaded = self._runtimes[package_id]
            except KeyError as exc:
                raise PluginWorkerError(f"plugin {package_id!r} is not loaded") from exc
            return loaded.execute(node_id, node_payload, cancellation)
        raise ValueError(f"unknown plugin worker operation: {operation}")

    def unload(self) -> Mapping[str, Any]:
        released = tuple(
            self._unload_one(package_id) for package_id in tuple(self._runtimes)
        )
        return {"released": len(released), "plugins": released}


def create_runtime() -> PluginMultiplexerRuntime:
    return PluginMultiplexerRuntime()


class SupervisorPluginBackend(PluginWorkerBackend):
    def __init__(
        self,
        supervisor: WorkerSupervisor,
        *,
        plugin_root: Path | str,
        permission_roots: Mapping[str, Path | str] | None = None,
        thread_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.plugin_root = Path(plugin_root).resolve(strict=True)
        if not self.plugin_root.is_dir():
            raise ValueError("plugin_root must be a directory")
        self.permission_roots = dict(permission_roots or {})
        self.thread_environment = dict(thread_environment or {})
        self._permissions: dict[str, PermissionSet] = {}

    @staticmethod
    def _worker_id(group_id: str) -> str:
        return f"plugin-{sha256(group_id.encode('utf-8')).hexdigest()[:16]}"

    async def start_group(self, group_id: str, permissions: PermissionSet) -> str:
        worker_id = self._worker_id(group_id)
        sandbox = SandboxPolicy.from_permissions(
            permissions,
            roots=self.permission_roots,
            working_directory=self.plugin_root,
            environment={},
        )
        spec = WorkerSpec(
            worker_id=worker_id,
            kind=WorkerKind.PLUGIN,
            entrypoint="comfyng.plugins.worker:PluginMultiplexerRuntime",
            sandbox=sandbox,
            thread_environment=self.thread_environment,
        )
        await asyncio.to_thread(self.supervisor.start, spec)
        self._permissions[worker_id] = permissions
        return worker_id

    async def load_plugin(self, worker_id: str, spec: RuntimePluginSpec) -> None:
        await asyncio.to_thread(
            self.supervisor.execute,
            worker_id,
            "plugin.load",
            {
                "package_id": spec.package_id,
                "version": spec.version,
                "install_path": str(spec.install_path),
                "entrypoint": spec.entrypoint,
            },
        )

    async def execute(
        self,
        worker_id: str,
        package_id: str,
        node_id: str,
        payload: object,
    ) -> object:
        if not isinstance(payload, Mapping):
            raise TypeError("plugin execution payload must be a mapping")
        return await asyncio.to_thread(
            self.supervisor.execute,
            worker_id,
            "plugin.execute",
            {"package_id": package_id, "node_id": node_id, "payload": dict(payload)},
        )

    async def unload_plugin(self, worker_id: str, package_id: str) -> None:
        await asyncio.to_thread(
            self.supervisor.execute,
            worker_id,
            "plugin.unload",
            {"package_id": package_id},
        )

    async def stop_group(self, worker_id: str) -> None:
        await asyncio.to_thread(self.supervisor.stop, worker_id)
        self._permissions.pop(worker_id, None)


__all__ = [
    "PluginMultiplexerRuntime",
    "PluginWorkerError",
    "SupervisorPluginBackend",
    "create_runtime",
]
