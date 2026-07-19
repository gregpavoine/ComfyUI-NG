from __future__ import annotations

import os
import sys
import time

import pytest

from comfyng.workers.protocol import (
    MAX_FRAME_BYTES,
    FrameError,
    WorkerCommand,
    WorkerCommandKind,
    WorkerEvent,
    WorkerEventKind,
    WorkerKind,
    WorkerSpec,
    decode_frame,
    encode_frame,
)
from comfyng.workers.process import select_start_method
from comfyng.workers.sandbox import SandboxPolicy
from comfyng.workers.supervisor import WorkerSupervisor
from comfyng.workers.supervisor import WorkerCrashedError


def test_protocol_is_a_length_delimited_msgspec_round_trip() -> None:
    command = WorkerCommand(
        command_id="command-1",
        kind=WorkerCommandKind.EXECUTE,
        operation="echo",
        payload={"text": "été", "items": [1, 2, 3]},
    )

    frame = encode_frame(command)

    assert int.from_bytes(frame[:4], "big") == len(frame) - 4
    assert decode_frame(frame, WorkerCommand) == command


@pytest.mark.parametrize("frame", (b"", b"\x00\x00\x00", b"\x00\x00\x00\x05abc"))
def test_protocol_rejects_truncated_frames(frame: bytes) -> None:
    with pytest.raises(FrameError, match="frame"):
        decode_frame(frame, WorkerCommand)


def test_protocol_rejects_a_different_message_type() -> None:
    event = WorkerEvent(
        worker_id="worker-1",
        kind=WorkerEventKind.READY,
        timestamp=1.0,
    )

    with pytest.raises(FrameError, match="WorkerCommand"):
        decode_frame(encode_frame(event), WorkerCommand)


def test_protocol_refuses_large_inline_payloads() -> None:
    command = WorkerCommand(
        command_id="oversized-command",
        kind=WorkerCommandKind.EXECUTE,
        operation="echo",
        payload={"forbidden_copy": b"x" * MAX_FRAME_BYTES},
    )

    with pytest.raises(FrameError, match="exceeds"):
        encode_frame(command)


@pytest.mark.parametrize(
    ("platform_name", "kind", "requested", "expected"),
    (
        ("darwin", WorkerKind.CPU_COMPUTE, "auto", "spawn"),
        ("darwin", WorkerKind.GPU_MODEL, "forkserver", "spawn"),
        ("linux", WorkerKind.CPU_COMPUTE, "auto", "forkserver"),
        ("linux", WorkerKind.PLUGIN, "auto", "forkserver"),
        ("linux", WorkerKind.GPU_MODEL, "auto", "spawn"),
        ("linux", WorkerKind.GPU_AUX, "forkserver", "spawn"),
    ),
)
def test_start_method_is_safe_for_platform_and_cuda(
    platform_name: str,
    kind: WorkerKind,
    requested: str,
    expected: str,
) -> None:
    assert (
        select_start_method(
            kind,
            requested=requested,
            platform_name=platform_name,
            available_methods=("fork", "spawn", "forkserver"),
        )
        == expected
    )


def test_fork_is_never_accepted() -> None:
    with pytest.raises(ValueError, match="fork"):
        select_start_method(
            WorkerKind.CPU_LIGHT,
            requested="fork",
            platform_name="linux",
            available_methods=("fork", "spawn", "forkserver"),
        )


def test_worker_starts_reports_heartbeats_and_executes() -> None:
    spec = WorkerSpec(
        worker_id="spawn-worker",
        kind=WorkerKind.CPU_LIGHT,
        heartbeat_interval=0.05,
        heartbeat_timeout=0.5,
    )
    with WorkerSupervisor() as supervisor:
        snapshot = supervisor.start(spec)
        first_heartbeat = snapshot.last_heartbeat

        assert snapshot.pid > 0
        assert snapshot.start_method == (
            "spawn" if sys.platform == "darwin" else "forkserver"
        )
        assert supervisor.execute(spec.worker_id, "echo", {"value": "ok"}) == {
            "value": "ok"
        }

        deadline = time.monotonic() + 1.0
        while supervisor.snapshot(spec.worker_id).last_heartbeat <= first_heartbeat:
            assert time.monotonic() < deadline
            time.sleep(0.02)


def test_worker_receives_a_bounded_environment() -> None:
    marker = "COMFYNG_TEST_PARENT_SECRET"
    os.environ[marker] = "must-not-leak"
    spec = WorkerSpec(
        worker_id="environment-worker",
        kind=WorkerKind.PLUGIN,
        sandbox=SandboxPolicy(
            inherit_environment=False,
            environment={"COMFYNG_ALLOWED": "yes"},
        ),
    )
    try:
        with WorkerSupervisor() as supervisor:
            supervisor.start(spec)
            environment = supervisor.execute(
                spec.worker_id,
                "environment",
                {"keys": [marker, "COMFYNG_ALLOWED"]},
            )
    finally:
        os.environ.pop(marker, None)

    assert environment == {marker: None, "COMFYNG_ALLOWED": "yes"}


def test_startup_failure_is_reported_as_a_worker_crash() -> None:
    spec = WorkerSpec(
        worker_id="broken-entrypoint",
        kind=WorkerKind.PLUGIN,
        entrypoint="comfyng.runtime.does_not_exist:Runtime",
        startup_timeout=5.0,
    )

    with WorkerSupervisor() as supervisor:
        with pytest.raises(WorkerCrashedError, match="startup"):
            supervisor.start(spec)


@pytest.mark.parametrize(
    ("entrypoint", "expected"),
    (
        ("comfyng.runtime.entrypoint:create_default_runtime", {"factory": True}),
        ("comfyng.runtime.entrypoint:diagnostic_handler", {"handler": True}),
    ),
)
def test_runtime_entrypoint_supports_factories_and_direct_handlers(
    entrypoint: str,
    expected: dict[str, bool],
) -> None:
    spec = WorkerSpec(
        worker_id=f"entrypoint-{next(iter(expected))}",
        kind=WorkerKind.CPU_LIGHT,
        entrypoint=entrypoint,
    )

    with WorkerSupervisor() as supervisor:
        supervisor.start(spec)
        assert supervisor.execute(spec.worker_id, "echo", expected) == expected
