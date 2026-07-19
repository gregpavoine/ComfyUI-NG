from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from uuid import uuid4

import msgspec


class SharedObjectError(RuntimeError):
    """Base error for managed shared-memory objects."""


class InvalidSharedObjectError(SharedObjectError):
    """Raised for forged, stale or otherwise unmanaged handles."""


class SharedObjectHandle(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    object_id: str
    name: str
    byte_size: int
    owner_worker: str
    content_type: str = "application/octet-stream"
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.object_id or not self.name or not self.owner_worker:
            raise ValueError("shared object identifiers must not be empty")
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")
        if self.checksum and len(self.checksum) != 64:
            raise ValueError("checksum must be a SHA-256 hexadecimal digest")


@dataclass(slots=True)
class _Allocation:
    handle: SharedObjectHandle
    leases: set[str] = field(default_factory=set)


class SharedObjectStore:
    """Owns named shared-memory segments and explicit worker leases."""

    def __init__(self) -> None:
        self._allocations: dict[str, _Allocation] = {}
        self._lock = threading.RLock()
        self._closed = False

    def __enter__(self) -> SharedObjectStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return sum(item.handle.byte_size for item in self._allocations.values())

    @property
    def object_count(self) -> int:
        with self._lock:
            return len(self._allocations)

    def create(
        self,
        owner_worker: str,
        data: bytes | bytearray | memoryview,
        *,
        content_type: str = "application/octet-stream",
    ) -> SharedObjectHandle:
        payload = bytes(data)
        if not owner_worker:
            raise ValueError("owner_worker must not be empty")
        with self._lock:
            if self._closed:
                raise SharedObjectError("shared object store is closed")
            segment = shared_memory.SharedMemory(create=True, size=max(1, len(payload)))
            try:
                if payload:
                    segment.buf[: len(payload)] = payload
                handle = SharedObjectHandle(
                    object_id=uuid4().hex,
                    name=segment.name,
                    byte_size=len(payload),
                    owner_worker=owner_worker,
                    content_type=content_type,
                    checksum=hashlib.sha256(payload).hexdigest(),
                )
                self._allocations[handle.object_id] = _Allocation(
                    handle=handle,
                    leases={owner_worker},
                )
            except BaseException:
                segment.unlink()
                raise
            finally:
                segment.close()
            return handle

    def _resolve(self, handle: SharedObjectHandle) -> _Allocation:
        allocation = self._allocations.get(handle.object_id)
        if allocation is None or allocation.handle != handle:
            raise InvalidSharedObjectError("shared object handle is not registered")
        return allocation

    def read(
        self, handle: SharedObjectHandle, *, verify_checksum: bool = False
    ) -> bytes:
        with self._lock:
            allocation = self._resolve(handle)
            segment = shared_memory.SharedMemory(
                name=allocation.handle.name,
                create=False,
                track=False,
            )
            try:
                payload = bytes(segment.buf[: allocation.handle.byte_size])
            finally:
                segment.close()
            if (
                verify_checksum
                and hashlib.sha256(payload).hexdigest() != handle.checksum
            ):
                raise InvalidSharedObjectError("shared object checksum mismatch")
            return payload

    def acquire(self, handle: SharedObjectHandle, worker_id: str) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        with self._lock:
            self._resolve(handle).leases.add(worker_id)

    def release(self, handle: SharedObjectHandle, worker_id: str) -> bool:
        with self._lock:
            allocation = self._resolve(handle)
            allocation.leases.discard(worker_id)
            if allocation.leases:
                return False
            self._reclaim(allocation)
            return True

    def cleanup_owner(self, owner_worker: str) -> int:
        """Force-revoke every object owned by a dead or stopped worker."""

        with self._lock:
            owned = [
                allocation
                for allocation in self._allocations.values()
                if allocation.handle.owner_worker == owner_worker
            ]
            reclaimed = sum(allocation.handle.byte_size for allocation in owned)
            for allocation in owned:
                self._reclaim(allocation)
            return reclaimed

    def _reclaim(self, allocation: _Allocation) -> None:
        self._allocations.pop(allocation.handle.object_id, None)
        try:
            segment = shared_memory.SharedMemory(
                name=allocation.handle.name,
                create=False,
                track=False,
            )
        except FileNotFoundError:
            return
        try:
            segment.unlink()
        finally:
            segment.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for allocation in tuple(self._allocations.values()):
                self._reclaim(allocation)
            self._closed = True
