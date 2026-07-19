"""Typed, isolated worker processes managed by the NG control plane.

Supervisor exports are lazy so a ``spawn`` child can import the protocol without
recursively importing the multiprocessing target that is currently unpickling.
"""

from typing import Any

from .protocol import (
    WorkerCommand,
    WorkerCommandKind,
    WorkerEvent,
    WorkerEventKind,
    WorkerKind,
    WorkerSpec,
)
from .shared_memory import SharedObjectHandle, SharedObjectStore

__all__ = [
    "SharedObjectHandle",
    "SharedObjectStore",
    "WorkerCommand",
    "WorkerCommandKind",
    "WorkerEvent",
    "WorkerEventKind",
    "WorkerKind",
    "WorkerSnapshot",
    "WorkerSpec",
    "WorkerSupervisor",
]


def __getattr__(name: str) -> Any:
    if name in {"WorkerSnapshot", "WorkerSupervisor"}:
        from .supervisor import WorkerSnapshot, WorkerSupervisor

        return {
            "WorkerSnapshot": WorkerSnapshot,
            "WorkerSupervisor": WorkerSupervisor,
        }[name]
    raise AttributeError(name)
