from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Protocol

from comfyng.core.enums import (
    LifecycleState,
    LoadPolicy,
    RuntimeIsolation,
    UnloadPolicy,
)
from comfyng.plugins.manifest import PluginManifest
from comfyng.plugins.permissions import PermissionSet


class PluginRuntimeError(RuntimeError):
    pass


class UnknownPluginError(PluginRuntimeError):
    pass


class PluginBusyError(PluginRuntimeError):
    pass


class TrustGroupPermissionMismatch(PluginRuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimePluginSpec:
    package_id: str
    version: str
    install_path: Path
    entrypoint: str
    isolation: RuntimeIsolation
    permissions: PermissionSet

    def __post_init__(self) -> None:
        if not self.package_id or not self.version:
            raise ValueError("runtime plugin package and version must be non-empty")
        if (
            not isinstance(self.install_path, Path)
            or not self.install_path.is_absolute()
        ):
            raise ValueError("runtime plugin install_path must be absolute")
        if ":" not in self.entrypoint:
            raise ValueError("runtime plugin entrypoint must contain ':'")
        if not isinstance(self.isolation, RuntimeIsolation):
            raise ValueError("runtime plugin isolation is invalid")
        if not isinstance(self.permissions, PermissionSet):
            raise ValueError("runtime plugin permissions are invalid")


class PluginWorkerBackend(Protocol):
    async def start_group(self, group_id: str, permissions: PermissionSet) -> str: ...

    async def load_plugin(self, worker_id: str, spec: RuntimePluginSpec) -> None: ...

    async def execute(
        self,
        worker_id: str,
        package_id: str,
        node_id: str,
        payload: object,
    ) -> object: ...

    async def unload_plugin(self, worker_id: str, package_id: str) -> None: ...

    async def stop_group(self, worker_id: str) -> None: ...


@dataclass(slots=True)
class RuntimePluginRecord:
    manifest: PluginManifest
    install_path: Path
    permissions: PermissionSet
    group_id: str
    state: LifecycleState = LifecycleState.DISCOVERED
    states: list[LifecycleState] = field(
        default_factory=lambda: [LifecycleState.DISCOVERED]
    )
    worker_id: str | None = None
    last_idle_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def transition(self, target: LifecycleState) -> None:
        self.state = self.state.transition_to(target)
        self.states.append(self.state)

    @property
    def spec(self) -> RuntimePluginSpec:
        return RuntimePluginSpec(
            package_id=self.manifest.package.id,
            version=self.manifest.package.version,
            install_path=self.install_path,
            entrypoint=self.manifest.runtime.entrypoint,
            isolation=self.manifest.runtime.isolation,
            permissions=self.permissions,
        )


@dataclass(slots=True)
class _WorkerGroup:
    group_id: str
    worker_id: str
    permissions: PermissionSet
    packages: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class PluginRuntimeManager:
    def __init__(
        self,
        backend: PluginWorkerBackend,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.clock = clock
        self._records: dict[str, RuntimePluginRecord] = {}
        self._groups: dict[str, _WorkerGroup] = {}
        self._manager_lock = asyncio.Lock()

    def register(
        self,
        manifest: PluginManifest,
        install_path: Path,
        permissions: PermissionSet,
        *,
        trust_group: str | None = None,
    ) -> RuntimePluginRecord:
        if not isinstance(manifest, PluginManifest):
            raise TypeError("manifest must be a PluginManifest")
        path = Path(install_path).resolve()
        if not path.is_absolute():
            raise ValueError("install_path must be absolute")
        if not isinstance(permissions, PermissionSet):
            raise TypeError("permissions must be a PermissionSet")
        package_id = manifest.package.id
        if package_id in self._records:
            raise ValueError(f"plugin {package_id!r} is already registered")
        if trust_group is not None:
            if not isinstance(trust_group, str) or not trust_group.strip():
                raise ValueError("trust_group must be a non-empty string")
            group_id = f"trust:{trust_group}"
        else:
            group_id = f"plugin:{package_id}"
        record = RuntimePluginRecord(
            manifest=manifest,
            install_path=path,
            permissions=permissions,
            group_id=group_id,
        )
        self._records[package_id] = record
        return record

    def record(self, package_id: str) -> RuntimePluginRecord:
        try:
            return self._records[package_id]
        except KeyError as exc:
            raise UnknownPluginError(package_id) from exc

    def state(self, package_id: str) -> LifecycleState:
        return self.record(package_id).state

    def history(self, package_id: str) -> tuple[LifecycleState, ...]:
        return tuple(self.record(package_id).states)

    async def _acquire_group(self, record: RuntimePluginRecord) -> _WorkerGroup:
        async with self._manager_lock:
            existing = self._groups.get(record.group_id)
            if existing is not None:
                if existing.permissions != record.permissions:
                    raise TrustGroupPermissionMismatch(
                        f"plugins sharing {record.group_id!r} require identical permissions"
                    )
                return existing
            worker_id = await self.backend.start_group(
                record.group_id,
                record.permissions,
            )
            group = _WorkerGroup(
                group_id=record.group_id,
                worker_id=worker_id,
                permissions=record.permissions,
            )
            self._groups[record.group_id] = group
            return group

    async def load(self, package_id: str) -> RuntimePluginRecord:
        record = self.record(package_id)
        async with record.lock:
            if record.state in (
                LifecycleState.READY,
                LifecycleState.IDLE,
                LifecycleState.BUSY,
            ):
                return record
            if record.state is LifecycleState.DISCOVERED:
                record.transition(LifecycleState.RESOLVED)
            elif record.state in (LifecycleState.UNLOADED, LifecycleState.FAILED):
                record.transition(LifecycleState.RESOLVED)
            if record.state is not LifecycleState.RESOLVED:
                raise PluginRuntimeError(
                    f"plugin {package_id} cannot load from {record.state.value}"
                )
            record.transition(LifecycleState.PRELOADING)
            group: _WorkerGroup | None = None
            try:
                group = await self._acquire_group(record)
                async with group.lock:
                    await self.backend.load_plugin(group.worker_id, record.spec)
                    group.packages.add(package_id)
                record.worker_id = group.worker_id
                record.transition(LifecycleState.LOADED)
                record.transition(LifecycleState.READY)
                return record
            except Exception:
                record.transition(LifecycleState.FAILED)
                if group is not None and not group.packages:
                    await self.backend.stop_group(group.worker_id)
                    self._groups.pop(group.group_id, None)
                raise

    async def _fail_group(self, record: RuntimePluginRecord) -> None:
        group = self._groups.pop(record.group_id, None)
        if group is None:
            return
        try:
            await self.backend.stop_group(group.worker_id)
        finally:
            for package_id in tuple(group.packages):
                affected = self._records[package_id]
                affected.worker_id = None
                if affected.state not in (
                    LifecycleState.FAILED,
                    LifecycleState.DISCOVERED,
                    LifecycleState.UNLOADED,
                ) and affected.state.can_transition_to(LifecycleState.FAILED):
                    affected.transition(LifecycleState.FAILED)

    async def execute(
        self,
        package_id: str,
        node_id: str,
        payload: object,
    ) -> Any:
        await self.load(package_id)
        record = self.record(package_id)
        async with record.lock:
            if record.state not in (LifecycleState.READY, LifecycleState.IDLE):
                raise PluginBusyError(
                    f"plugin {package_id} is not executable from {record.state.value}"
                )
            if record.worker_id is None:
                raise PluginRuntimeError(f"plugin {package_id} has no worker")
            group = self._groups.get(record.group_id)
            if group is None or group.worker_id != record.worker_id:
                raise PluginRuntimeError(f"plugin {package_id} worker group is unavailable")
            record.transition(LifecycleState.BUSY)
            try:
                async with group.lock:
                    result = await self.backend.execute(
                        record.worker_id,
                        package_id,
                        node_id,
                        payload,
                    )
            except Exception:
                record.transition(LifecycleState.FAILED)
                await self._fail_group(record)
                raise
            record.transition(LifecycleState.IDLE)
            record.last_idle_at = self.clock()
        if record.manifest.runtime.unload_policy is UnloadPolicy.UNLOAD_AFTER_EXECUTION:
            await self.unload(package_id)
        return result

    async def unload(self, package_id: str) -> bool:
        record = self.record(package_id)
        if record.state is LifecycleState.BUSY:
            raise PluginBusyError(f"plugin {package_id} is busy")
        async with record.lock:
            if record.state in (LifecycleState.DISCOVERED, LifecycleState.UNLOADED):
                return False
            if record.state not in (
                LifecycleState.READY,
                LifecycleState.IDLE,
                LifecycleState.FAILED,
            ):
                raise PluginBusyError(
                    f"plugin {package_id} cannot unload from {record.state.value}"
                )
            record.transition(LifecycleState.EVICTING)
            worker_id = record.worker_id
            group = self._groups.get(record.group_id)
            try:
                if group is None:
                    if worker_id is not None:
                        await self.backend.unload_plugin(worker_id, package_id)
                else:
                    async with group.lock:
                        if worker_id is not None:
                            await self.backend.unload_plugin(worker_id, package_id)
                        group.packages.discard(package_id)
                        if not group.packages:
                            await self.backend.stop_group(group.worker_id)
                            self._groups.pop(group.group_id, None)
                record.worker_id = None
                record.transition(LifecycleState.UNLOADED)
                return True
            except Exception:
                record.transition(LifecycleState.FAILED)
                raise

    async def _preload(
        self,
        package_ids: Iterable[str],
        policy: LoadPolicy,
    ) -> tuple[str, ...]:
        loaded: list[str] = []
        for package_id in package_ids:
            record = self.record(package_id)
            if record.manifest.runtime.load_policy is policy:
                await self.load(package_id)
                loaded.append(package_id)
        return tuple(loaded)

    async def preload_for_workflow(
        self,
        package_ids: Iterable[str],
    ) -> tuple[str, ...]:
        return await self._preload(package_ids, LoadPolicy.PRELOAD_ON_WORKFLOW_OPEN)

    async def preload_for_queue(
        self,
        package_ids: Iterable[str],
    ) -> tuple[str, ...]:
        return await self._preload(package_ids, LoadPolicy.PRELOAD_ON_QUEUE)

    async def preload_keep_warm(
        self,
        package_ids: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        selected = self._records if package_ids is None else package_ids
        return await self._preload(selected, LoadPolicy.KEEP_WARM)

    async def sweep_idle(self) -> tuple[str, ...]:
        now = self.clock()
        unloaded: list[str] = []
        for package_id, record in tuple(self._records.items()):
            runtime = record.manifest.runtime
            if (
                runtime.unload_policy is UnloadPolicy.UNLOAD_AFTER_IDLE
                and record.state is LifecycleState.IDLE
                and record.last_idle_at is not None
                and now - record.last_idle_at >= runtime.idle_timeout_seconds
            ):
                await self.unload(package_id)
                unloaded.append(package_id)
        return tuple(unloaded)

    async def handle_memory_pressure(self) -> tuple[str, ...]:
        unloaded: list[str] = []
        for package_id, record in tuple(self._records.items()):
            if (
                record.manifest.runtime.unload_policy
                is UnloadPolicy.UNLOAD_ON_MEMORY_PRESSURE
                and record.state in (LifecycleState.READY, LifecycleState.IDLE)
            ):
                await self.unload(package_id)
                unloaded.append(package_id)
        return tuple(unloaded)


__all__ = [
    "PluginBusyError",
    "PluginRuntimeError",
    "PluginRuntimeManager",
    "PluginWorkerBackend",
    "RuntimePluginRecord",
    "RuntimePluginSpec",
    "TrustGroupPermissionMismatch",
    "UnknownPluginError",
]
