from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from comfyng.workers.protocol import WorkerKind, WorkerSpec
from comfyng.workers.sandbox import SandboxPolicy
from comfyng.workers.supervisor import (
    RemoteExecutionError,
    WorkerCancelledError,
    WorkerCrashedError,
    WorkerSupervisor,
    WorkerUnavailableError,
)


def test_unload_releases_runtime_state_without_restarting_worker() -> None:
    spec = WorkerSpec(worker_id="unload-worker", kind=WorkerKind.GPU_MODEL)
    with WorkerSupervisor() as supervisor:
        pid = supervisor.start(spec).pid
        supervisor.execute(spec.worker_id, "load_resource", {"name": "transformer"})
        supervisor.execute(spec.worker_id, "load_resource", {"name": "vae"})

        assert supervisor.unload(spec.worker_id) == {"released": 2}
        assert supervisor.snapshot(spec.worker_id).pid == pid
        assert supervisor.execute(spec.worker_id, "resource_count") == {"count": 0}


def test_graceful_stop_is_idempotent_and_rejects_future_work() -> None:
    spec = WorkerSpec(worker_id="stop-worker", kind=WorkerKind.IO)
    supervisor = WorkerSupervisor()
    supervisor.start(spec)

    supervisor.stop(spec.worker_id)
    supervisor.stop(spec.worker_id)

    assert supervisor.snapshot(spec.worker_id).status == "stopped"
    with pytest.raises(WorkerUnavailableError, match="stop-worker"):
        supervisor.execute(spec.worker_id, "echo", {"value": 1})
    supervisor.close()


def test_context_manager_reclaims_every_worker() -> None:
    specs = (
        WorkerSpec(worker_id="close-a", kind=WorkerKind.CPU_LIGHT),
        WorkerSpec(worker_id="close-b", kind=WorkerKind.IO),
    )
    supervisor = WorkerSupervisor()
    with supervisor:
        pids = [supervisor.start(spec).pid for spec in specs]

    deadline = time.monotonic() + 3.0
    for pid in pids:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"worker PID {pid} was not reclaimed")


def test_plugin_sandbox_blocks_network_connections() -> None:
    spec = WorkerSpec(
        worker_id="network-sandbox",
        kind=WorkerKind.PLUGIN,
        sandbox=SandboxPolicy(allow_network=False),
    )
    with WorkerSupervisor() as supervisor:
        supervisor.start(spec)

        with pytest.raises(RemoteExecutionError, match="network access denied"):
            supervisor.execute(
                spec.worker_id,
                "connect",
                {"host": "127.0.0.1", "port": 9},
            )


def test_plugin_sandbox_blocks_subprocesses_by_default() -> None:
    spec = WorkerSpec(worker_id="process-sandbox", kind=WorkerKind.PLUGIN)
    with WorkerSupervisor() as supervisor:
        supervisor.start(spec)

        with pytest.raises(RemoteExecutionError, match="subprocess creation denied"):
            supervisor.execute(spec.worker_id, "spawn_child", {"seconds": 1})


@pytest.mark.skipif(os.name != "posix", reason="POSIX process primitives only")
@pytest.mark.parametrize(
    "operation",
    ("fork_process", "forkpty_process", "exec_process", "pty_spawn"),
)
def test_plugin_sandbox_blocks_direct_process_primitive_bypasses(
    operation: str,
) -> None:
    spec = WorkerSpec(worker_id=f"blocked-{operation}", kind=WorkerKind.PLUGIN)
    with WorkerSupervisor() as supervisor:
        supervisor.start(spec)

        with pytest.raises(RemoteExecutionError, match="subprocess creation denied"):
            supervisor.execute(spec.worker_id, operation)
        assert supervisor.execute(spec.worker_id, "echo", {"alive": True}) == {
            "alive": True
        }


def test_explicit_cancellation_keeps_a_cooperative_worker_alive() -> None:
    spec = WorkerSpec(worker_id="cancel-worker", kind=WorkerKind.CPU_LIGHT)
    outcome: list[BaseException] = []
    with WorkerSupervisor() as supervisor:
        pid = supervisor.start(spec).pid

        def execute() -> None:
            try:
                supervisor.execute(
                    spec.worker_id,
                    "sleep",
                    {"seconds": 5.0},
                    timeout=10.0,
                )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=execute)
        thread.start()
        deadline = time.monotonic() + 2.0
        while not supervisor.snapshot(spec.worker_id).busy:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        assert supervisor.cancel(spec.worker_id) is True
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], WorkerCancelledError)
        assert supervisor.snapshot(spec.worker_id).pid == pid
        assert supervisor.execute(spec.worker_id, "echo", {"ready": True}) == {
            "ready": True
        }


def test_worker_spec_rejects_unsafe_timeouts_and_restart_budgets() -> None:
    with pytest.raises(ValueError, match="heartbeat_interval"):
        WorkerSpec(
            worker_id="invalid",
            kind=WorkerKind.IO,
            heartbeat_interval=1.0,
            heartbeat_timeout=0.5,
        )
    with pytest.raises(ValueError, match="restart_limit"):
        WorkerSpec(worker_id="invalid", kind=WorkerKind.IO, restart_limit=-1)


def _filesystem_spec(
    tmp_path: Path,
    *,
    read_roots: tuple[Path, ...] = (),
    write_roots: tuple[Path, ...] = (),
) -> WorkerSpec:
    return WorkerSpec(
        worker_id=f"filesystem-{len(read_roots)}-{len(write_roots)}",
        kind=WorkerKind.PLUGIN,
        sandbox=SandboxPolicy(
            filesystem_read_roots=tuple(map(str, read_roots)),
            filesystem_write_roots=tuple(map(str, write_roots)),
        ),
    )


def test_filesystem_sandbox_allows_only_declared_read_and_write_roots(
    tmp_path: Path,
) -> None:
    readable = tmp_path / "readable"
    writable = tmp_path / "writable"
    outside = tmp_path / "outside"
    for directory in (readable, writable, outside):
        directory.mkdir()
    (readable / "allowed.txt").write_text("allowed", encoding="utf-8")
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    spec = _filesystem_spec(
        tmp_path,
        read_roots=(readable,),
        write_roots=(writable,),
    )

    with WorkerSupervisor() as supervisor:
        supervisor.start(spec)
        assert supervisor.execute(
            spec.worker_id, "read_text", {"path": str(readable / "allowed.txt")}
        ) == {"text": "allowed"}
        assert supervisor.execute(
            spec.worker_id,
            "write_text",
            {"path": str(writable / "created.txt"), "text": "created"},
        ) == {"bytes": 7}
        with pytest.raises(RemoteExecutionError, match="filesystem read denied"):
            supervisor.execute(
                spec.worker_id, "read_text", {"path": str(outside / "secret.txt")}
            )
        with pytest.raises(RemoteExecutionError, match="filesystem write denied"):
            supervisor.execute(
                spec.worker_id,
                "write_text",
                {"path": str(outside / "stolen.txt"), "text": "denied"},
            )

    assert (writable / "created.txt").read_text(encoding="utf-8") == "created"
    assert not (outside / "stolen.txt").exists()


def test_filesystem_sandbox_denies_parent_traversal_and_symlink_escape(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (allowed / "rename-me.txt").write_text("content", encoding="utf-8")
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    spec = _filesystem_spec(
        tmp_path,
        read_roots=(allowed,),
        write_roots=(allowed,),
    )

    with WorkerSupervisor() as supervisor:
        supervisor.start(spec)
        for path in (
            allowed / ".." / "outside" / "secret.txt",
            allowed / "escape" / "secret.txt",
        ):
            with pytest.raises(RemoteExecutionError, match="filesystem read denied"):
                supervisor.execute(spec.worker_id, "read_text", {"path": str(path)})
        for path in (
            allowed / ".." / "outside" / "created.txt",
            allowed / "escape" / "created.txt",
        ):
            with pytest.raises(RemoteExecutionError, match="filesystem write denied"):
                supervisor.execute(
                    spec.worker_id,
                    "write_text",
                    {"path": str(path), "text": "denied"},
                )
        with pytest.raises(RemoteExecutionError, match="filesystem write denied"):
            supervisor.execute(
                spec.worker_id,
                "rename_path",
                {
                    "source": str(allowed / "rename-me.txt"),
                    "destination": str(outside / "renamed.txt"),
                },
            )
        with pytest.raises(RemoteExecutionError, match="symlink target denied"):
            supervisor.execute(
                spec.worker_id,
                "create_symlink",
                {
                    "source": str(outside),
                    "destination": str(allowed / "malicious-link"),
                },
            )

    assert not (outside / "created.txt").exists()
    assert (allowed / "rename-me.txt").exists()
    assert not (outside / "renamed.txt").exists()
    assert not (allowed / "malicious-link").exists()


def test_filesystem_sandbox_is_active_before_plugin_module_import(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("must not be imported", encoding="utf-8")
    spec = WorkerSpec(
        worker_id="malicious-import",
        kind=WorkerKind.PLUGIN,
        entrypoint="tests.integration.workers.malicious_plugin:create_runtime",
        sandbox=SandboxPolicy(
            environment={"COMFYNG_MALICIOUS_READ": str(secret)},
        ),
    )

    with WorkerSupervisor() as supervisor:
        with pytest.raises(WorkerCrashedError, match="filesystem read denied"):
            supervisor.start(spec)


def test_sandbox_can_be_built_from_permission_aliases_without_plugin_import(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    output = tmp_path / "output"
    models.mkdir()
    output.mkdir()

    policy = SandboxPolicy.from_permissions(
        {
            "network": False,
            "subprocess": False,
            "filesystem_read": ("models",),
            "filesystem_write": ("output",),
        },
        roots={"models": models, "output": output},
    )

    assert policy.filesystem_read_roots == (str(models.resolve()),)
    assert policy.filesystem_write_roots == (str(output.resolve()),)
