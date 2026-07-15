from __future__ import annotations

from enum import StrEnum

from .errors import InvalidLifecycleTransition


class LifecycleState(StrEnum):
    DISCOVERED = "DISCOVERED"
    RESOLVED = "RESOLVED"
    PRELOADING = "PRELOADING"
    LOADED = "LOADED"
    READY = "READY"
    BUSY = "BUSY"
    IDLE = "IDLE"
    EVICTING = "EVICTING"
    UNLOADED = "UNLOADED"
    FAILED = "FAILED"

    def can_transition_to(self, target: LifecycleState | str) -> bool:
        try:
            resolved = LifecycleState(target)
        except ValueError:
            return False
        return resolved in _LIFECYCLE_TRANSITIONS[self]

    def transition_to(self, target: LifecycleState | str) -> LifecycleState:
        try:
            resolved = LifecycleState(target)
        except ValueError as exc:
            raise InvalidLifecycleTransition(self.value, target) from exc
        if not self.can_transition_to(resolved):
            raise InvalidLifecycleTransition(self.value, resolved.value)
        return resolved


_LIFECYCLE_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.DISCOVERED: frozenset(
        (LifecycleState.RESOLVED, LifecycleState.FAILED)
    ),
    LifecycleState.RESOLVED: frozenset(
        (LifecycleState.PRELOADING, LifecycleState.FAILED)
    ),
    LifecycleState.PRELOADING: frozenset(
        (LifecycleState.LOADED, LifecycleState.FAILED)
    ),
    LifecycleState.LOADED: frozenset(
        (LifecycleState.READY, LifecycleState.FAILED)
    ),
    LifecycleState.READY: frozenset(
        (
            LifecycleState.BUSY,
            LifecycleState.IDLE,
            LifecycleState.EVICTING,
            LifecycleState.FAILED,
        )
    ),
    LifecycleState.BUSY: frozenset(
        (LifecycleState.IDLE, LifecycleState.FAILED)
    ),
    LifecycleState.IDLE: frozenset(
        (LifecycleState.BUSY, LifecycleState.EVICTING, LifecycleState.FAILED)
    ),
    LifecycleState.EVICTING: frozenset(
        (LifecycleState.UNLOADED, LifecycleState.FAILED)
    ),
    LifecycleState.UNLOADED: frozenset((LifecycleState.RESOLVED,)),
    LifecycleState.FAILED: frozenset(
        (LifecycleState.RESOLVED, LifecycleState.EVICTING)
    ),
}


NodeLifecycleState = LifecycleState


class LoadPolicy(StrEnum):
    LOAD_ON_EXECUTION = "LOAD_ON_EXECUTION"
    PRELOAD_ON_WORKFLOW_OPEN = "PRELOAD_ON_WORKFLOW_OPEN"
    PRELOAD_ON_QUEUE = "PRELOAD_ON_QUEUE"
    KEEP_WARM = "KEEP_WARM"


class UnloadPolicy(StrEnum):
    UNLOAD_AFTER_EXECUTION = "UNLOAD_AFTER_EXECUTION"
    UNLOAD_AFTER_IDLE = "UNLOAD_AFTER_IDLE"
    UNLOAD_ON_MEMORY_PRESSURE = "UNLOAD_ON_MEMORY_PRESSURE"
    PERSISTENT = "PERSISTENT"


class TransferPolicy(StrEnum):
    INLINE = "inline"
    HANDLE = "handle"
    SAME_WORKER = "same_worker"
    SHARED_MEMORY = "shared_memory"
    TEMPORARY_FILE = "temporary_file"


class SerializationStrategy(StrEnum):
    JSON = "JSON"
    MSGSPEC = "MSGSPEC"
    SHARED_HANDLE = "SHARED_HANDLE"


class RuntimeIsolation(StrEnum):
    PLUGIN_WORKER = "plugin_worker"
    GPU_MODEL_WORKER = "gpu_model_worker"
    CPU_WORKER = "cpu_worker"
    IO_WORKER = "io_worker"
    ISOLATED_PROCESS = "isolated_process"


class GpuRequirement(StrEnum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"
