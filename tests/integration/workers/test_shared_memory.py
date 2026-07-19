from __future__ import annotations

from multiprocessing import shared_memory

import pytest

from comfyng.workers.protocol import WorkerKind, WorkerSpec
from comfyng.workers.shared_memory import (
    InvalidSharedObjectError,
    SharedObjectStore,
)
from comfyng.workers.supervisor import WorkerCrashedError, WorkerSupervisor


def test_shared_object_round_trip_uses_an_opaque_handle() -> None:
    with SharedObjectStore() as store:
        handle = store.create("worker-a", b"\x00\x01payload", content_type="bytes")

        assert handle.owner_worker == "worker-a"
        assert handle.byte_size == 9
        assert not hasattr(handle, "data")
        assert store.read(handle) == b"\x00\x01payload"
        assert store.read(handle, verify_checksum=True) == b"\x00\x01payload"
        assert store.allocated_bytes == 9


def test_shared_object_leases_delay_reclamation_until_all_lessees_release() -> None:
    with SharedObjectStore() as store:
        handle = store.create("owner", b"leased")
        store.acquire(handle, "consumer")

        assert store.release(handle, "owner") is False
        assert store.read(handle) == b"leased"
        assert store.release(handle, "consumer") is True
        assert store.allocated_bytes == 0
        with pytest.raises(InvalidSharedObjectError):
            store.read(handle)


def test_owner_cleanup_revokes_all_leases_and_unlinks_memory() -> None:
    with SharedObjectStore() as store:
        handle = store.create("owner", b"temporary")
        store.acquire(handle, "consumer")

        assert store.cleanup_owner("owner") == handle.byte_size
        assert store.allocated_bytes == 0
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.name, create=False)


def test_forged_handle_cannot_read_an_unregistered_segment() -> None:
    with SharedObjectStore() as store:
        handle = store.create("owner", b"authentic")
        forged = handle.__class__(
            object_id="forged",
            name=handle.name,
            byte_size=handle.byte_size,
            owner_worker=handle.owner_worker,
            content_type=handle.content_type,
            checksum=handle.checksum,
        )

        with pytest.raises(InvalidSharedObjectError, match="registered"):
            store.read(forged)


def test_worker_can_read_and_mutate_shared_memory_without_json_copy() -> None:
    store = SharedObjectStore()
    spec = WorkerSpec(worker_id="shared-worker", kind=WorkerKind.CPU_COMPUTE)
    with WorkerSupervisor(shared_store=store) as supervisor:
        supervisor.start(spec)
        handle = store.create(spec.worker_id, b"abcdef")

        result = supervisor.execute(
            spec.worker_id,
            "reverse_shared",
            {"name": handle.name, "byte_size": handle.byte_size},
        )

        assert result == {"byte_size": 6}
        assert store.read(handle) == b"fedcba"
        with pytest.raises(InvalidSharedObjectError, match="checksum"):
            store.read(handle, verify_checksum=True)


def test_worker_crash_reclaims_all_owned_shared_memory() -> None:
    store = SharedObjectStore()
    spec = WorkerSpec(worker_id="shared-crash", kind=WorkerKind.PLUGIN)
    with WorkerSupervisor(shared_store=store) as supervisor:
        supervisor.start(spec)
        handle = store.create(spec.worker_id, b"must-be-reclaimed")

        with pytest.raises(WorkerCrashedError):
            supervisor.execute(spec.worker_id, "crash", timeout=1.0)

        assert store.allocated_bytes == 0
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=handle.name, create=False)
