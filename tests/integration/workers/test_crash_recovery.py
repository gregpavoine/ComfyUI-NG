from __future__ import annotations

import os
import time

import pytest

from comfyng.workers.protocol import WorkerKind, WorkerSpec
from comfyng.workers.sandbox import SandboxPolicy
from comfyng.workers.supervisor import (
    CircuitOpenError,
    WorkerCrashedError,
    WorkerSupervisor,
    WorkerTimeoutError,
)


def _wait_for_new_pid(
    supervisor: WorkerSupervisor,
    worker_id: str,
    previous_pid: int,
    timeout: float = 10.0,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = supervisor.snapshot(worker_id)
        if snapshot.ready and snapshot.pid != previous_pid:
            return snapshot.pid
        time.sleep(0.02)
    raise AssertionError("worker was not restarted")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _pid_exists(pid):
        time.sleep(0.02)
    assert not _pid_exists(pid), f"PID {pid} survived worker cleanup"


def test_a_crashing_plugin_is_isolated_and_restarted() -> None:
    crashing = WorkerSpec(
        worker_id="crashing-plugin",
        kind=WorkerKind.PLUGIN,
        restart_limit=2,
        restart_window=10.0,
    )
    healthy = WorkerSpec(worker_id="healthy-worker", kind=WorkerKind.CPU_LIGHT)
    with WorkerSupervisor() as supervisor:
        old_pid = supervisor.start(crashing).pid
        healthy_pid = supervisor.start(healthy).pid

        with pytest.raises(WorkerCrashedError, match="crashing-plugin"):
            supervisor.execute(crashing.worker_id, "crash", timeout=1.0)

        new_pid = _wait_for_new_pid(supervisor, crashing.worker_id, old_pid)
        assert new_pid != old_pid
        assert supervisor.execute(crashing.worker_id, "echo", {"recovered": True}) == {
            "recovered": True
        }
        assert supervisor.snapshot(healthy.worker_id).pid == healthy_pid
        assert supervisor.execute(healthy.worker_id, "echo", {"alive": True}) == {
            "alive": True
        }


def test_restart_budget_opens_a_circuit_breaker() -> None:
    spec = WorkerSpec(
        worker_id="bounded-restarts",
        kind=WorkerKind.PLUGIN,
        restart_limit=1,
        restart_window=60.0,
    )
    with WorkerSupervisor() as supervisor:
        first_pid = supervisor.start(spec).pid
        with pytest.raises(WorkerCrashedError):
            supervisor.execute(spec.worker_id, "crash", timeout=1.0)
        _wait_for_new_pid(supervisor, spec.worker_id, first_pid)

        with pytest.raises(WorkerCrashedError):
            supervisor.execute(spec.worker_id, "crash", timeout=1.0)

        deadline = time.monotonic() + 2.0
        while not supervisor.snapshot(spec.worker_id).circuit_open:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        with pytest.raises(CircuitOpenError, match="bounded-restarts"):
            supervisor.execute(spec.worker_id, "echo", {"never": "runs"})


def test_execution_timeout_recycles_a_hung_worker() -> None:
    spec = WorkerSpec(
        worker_id="hung-operation",
        kind=WorkerKind.PLUGIN,
        cancellation_grace=0.1,
        restart_limit=2,
    )
    with WorkerSupervisor() as supervisor:
        old_pid = supervisor.start(spec).pid

        with pytest.raises(WorkerTimeoutError, match="hung-operation"):
            supervisor.execute(spec.worker_id, "hang", timeout=0.15)

        _wait_for_pid_exit(old_pid)
        _wait_for_new_pid(supervisor, spec.worker_id, old_pid)
        assert supervisor.execute(spec.worker_id, "echo", {"ready": True}) == {
            "ready": True
        }


@pytest.mark.skipif(os.name != "posix", reason="SIGSTOP watchdog test is POSIX-only")
def test_missing_heartbeats_are_detected_and_worker_is_restarted() -> None:
    spec = WorkerSpec(
        worker_id="stopped-worker",
        kind=WorkerKind.PLUGIN,
        heartbeat_interval=0.03,
        heartbeat_timeout=0.15,
        restart_limit=2,
    )
    with WorkerSupervisor() as supervisor:
        old_pid = supervisor.start(spec).pid
        with pytest.raises(WorkerCrashedError, match="heartbeat timeout"):
            supervisor.execute(spec.worker_id, "freeze", timeout=1.5)
        _wait_for_pid_exit(old_pid)
        _wait_for_new_pid(supervisor, spec.worker_id, old_pid)


def test_stop_terminates_the_complete_process_tree() -> None:
    spec = WorkerSpec(
        worker_id="tree-worker",
        kind=WorkerKind.PLUGIN,
        sandbox_allow_subprocess=True,
    )
    supervisor = WorkerSupervisor()
    worker_pid = supervisor.start(spec).pid
    child_pid = supervisor.execute(spec.worker_id, "spawn_child", {"seconds": 60})[
        "pid"
    ]
    assert _pid_exists(worker_pid)
    assert _pid_exists(child_pid)

    supervisor.stop(spec.worker_id)

    _wait_for_pid_exit(worker_pid)
    _wait_for_pid_exit(child_pid)


@pytest.mark.skipif(os.name != "posix", reason="process-group test is POSIX-only")
def test_watchdog_kills_descendants_of_a_frozen_worker(tmp_path) -> None:
    pid_path = tmp_path / "descendant.pid"
    spec = WorkerSpec(
        worker_id="frozen-tree-worker",
        kind=WorkerKind.PLUGIN,
        heartbeat_interval=0.03,
        heartbeat_timeout=0.15,
        restart_limit=0,
        sandbox_allow_subprocess=True,
        sandbox=SandboxPolicy(
            allow_subprocess=True,
            filesystem_write_roots=(str(tmp_path),),
        ),
    )
    supervisor = WorkerSupervisor()
    worker_pid = supervisor.start(spec).pid

    with pytest.raises(WorkerCrashedError, match="heartbeat timeout"):
        supervisor.execute(
            spec.worker_id,
            "spawn_child_and_freeze",
            {"pid_path": str(pid_path), "seconds": 60},
            timeout=2.0,
        )

    child_pid = int(pid_path.read_text(encoding="utf-8"))
    _wait_for_pid_exit(worker_pid)
    _wait_for_pid_exit(child_pid)
    supervisor.close()
