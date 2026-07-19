from __future__ import annotations

import importlib
import inspect
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol

from comfyng.workers.protocol import (
    MAX_FRAME_BYTES,
    WorkerCommand,
    WorkerCommandKind,
    WorkerEvent,
    WorkerEventKind,
    WorkerSpec,
    decode_frame,
    encode_frame,
)
from comfyng.workers.sandbox import SandboxPolicy, apply_sandbox, default_policy_for


class CancelledExecution(RuntimeError):
    """Internal marker used to return a typed cancellation event."""


class RuntimeHandler(Protocol):
    def execute(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any: ...

    def unload(self) -> Mapping[str, Any]: ...


class DefaultRuntime:
    """Small standard runtime used for diagnostics and worker plumbing."""

    def __init__(self) -> None:
        self._resources: set[str] = set()
        self._children: list[subprocess.Popen[bytes]] = []

    def execute(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        if operation == "echo":
            return dict(payload)
        if operation == "pid":
            return {"pid": os.getpid()}
        if operation == "environment":
            keys = payload.get("keys", ())
            if not isinstance(keys, (list, tuple)) or not all(
                isinstance(item, str) for item in keys
            ):
                raise ValueError("environment keys must be a string sequence")
            return {key: os.environ.get(key) for key in keys}
        if operation == "sleep":
            seconds = _positive_seconds(payload)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if cancellation.wait(min(0.02, max(0.0, deadline - time.monotonic()))):
                    raise CancelledExecution("execution cancelled")
            return {"slept": seconds}
        if operation == "hang":
            while True:
                time.sleep(60)
        if operation == "crash":
            os._exit(91)
        if operation == "freeze":
            if not hasattr(signal, "SIGSTOP"):
                raise RuntimeError("freeze is unsupported on this platform")
            os.kill(os.getpid(), signal.SIGSTOP)
            raise RuntimeError("worker unexpectedly resumed")
        if operation == "spawn_child":
            seconds = _positive_seconds(payload, default=60.0)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"import time; time.sleep({seconds!r})",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._children.append(child)
            return {"pid": child.pid}
        if operation == "spawn_child_and_freeze":
            seconds = _positive_seconds(payload, default=60.0)
            pid_path = payload.get("pid_path")
            if not isinstance(pid_path, str) or not pid_path:
                raise ValueError("spawn_child_and_freeze requires pid_path")
            if not hasattr(signal, "SIGSTOP"):
                raise RuntimeError("freeze is unsupported on this platform")
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"import time; time.sleep({seconds!r})",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._children.append(child)
            with Path(pid_path).open("w", encoding="utf-8") as stream:
                stream.write(str(child.pid))
                stream.flush()
                os.fsync(stream.fileno())
            os.kill(os.getpid(), signal.SIGSTOP)
            raise RuntimeError("worker unexpectedly resumed")
        if operation == "connect":
            host = payload.get("host")
            port = payload.get("port")
            if (
                not isinstance(host, str)
                or not isinstance(port, int)
                or isinstance(port, bool)
            ):
                raise ValueError("connect requires a host and integer port")
            with socket.create_connection((host, port), timeout=0.25):
                return {"connected": True}
        if operation == "read_text":
            path = payload.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError("read_text requires a path")
            return {"text": Path(path).read_text(encoding="utf-8")}
        if operation == "write_text":
            path = payload.get("path")
            text = payload.get("text")
            if not isinstance(path, str) or not isinstance(text, str):
                raise ValueError("write_text requires path and text strings")
            written = Path(path).write_text(text, encoding="utf-8")
            return {"bytes": written}
        if operation == "rename_path":
            source = payload.get("source")
            destination = payload.get("destination")
            if not isinstance(source, str) or not isinstance(destination, str):
                raise ValueError("rename_path requires source and destination strings")
            os.rename(source, destination)
            return {"renamed": True}
        if operation == "create_symlink":
            source = payload.get("source")
            destination = payload.get("destination")
            if not isinstance(source, str) or not isinstance(destination, str):
                raise ValueError(
                    "create_symlink requires source and destination strings"
                )
            os.symlink(source, destination)
            return {"created": True}
        if operation == "reverse_shared":
            name = payload.get("name")
            byte_size = payload.get("byte_size")
            if (
                not isinstance(name, str)
                or not isinstance(byte_size, int)
                or isinstance(byte_size, bool)
                or byte_size < 0
            ):
                raise ValueError("reverse_shared requires a valid name and byte_size")
            segment = shared_memory.SharedMemory(name=name, create=False, track=False)
            try:
                if byte_size > len(segment.buf):
                    raise ValueError("shared-memory handle exceeds the segment")
                value = bytes(segment.buf[:byte_size])
                segment.buf[:byte_size] = value[::-1]
            finally:
                segment.close()
            return {"byte_size": byte_size}
        if operation == "load_resource":
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("resource name must not be empty")
            self._resources.add(name)
            return {"loaded": name}
        if operation == "resource_count":
            return {"count": len(self._resources)}
        if operation == "fork_process":
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            return {"forked": pid}
        if operation == "forkpty_process":
            if not hasattr(os, "forkpty"):
                raise RuntimeError("forkpty is unsupported on this platform")
            pid, fd = os.forkpty()
            if pid == 0:
                os._exit(0)
            os.close(fd)
            os.waitpid(pid, 0)
            return {"forked": pid}
        if operation == "exec_process":
            os.execv(sys.executable, [sys.executable, "-c", "pass"])
        if operation == "pty_spawn":
            import pty

            pty.spawn([sys.executable, "-c", "pass"])
            return {"spawned": True}
        raise ValueError(f"unknown worker operation: {operation}")

    def unload(self) -> Mapping[str, Any]:
        released = len(self._resources)
        self._resources.clear()
        for child in tuple(self._children):
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=0.5)
        self._children.clear()
        return {"released": released}


class _FunctionRuntime:
    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function

    def execute(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        return self._function(operation, payload, cancellation)

    def unload(self) -> Mapping[str, Any]:
        return {"released": 0}


def create_default_runtime() -> DefaultRuntime:
    """Standard zero-argument runtime factory contract used by manifests."""

    return DefaultRuntime()


def diagnostic_handler(
    operation: str,
    payload: Mapping[str, Any],
    cancellation: threading.Event,
) -> Any:
    """Direct three-argument handler contract retained for lightweight workers."""

    return DefaultRuntime().execute(operation, payload, cancellation)


def _positive_seconds(
    payload: Mapping[str, Any],
    *,
    default: float | None = None,
) -> float:
    value = payload.get("seconds", default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError("seconds must be a positive number")
    return float(value)


def _load_runtime(entrypoint: str | None) -> RuntimeHandler:
    if entrypoint is None:
        return DefaultRuntime()
    module_name, attribute_name = entrypoint.split(":", 1)
    target = getattr(importlib.import_module(module_name), attribute_name)
    if inspect.isclass(target):
        runtime = target()
    elif callable(target) and not hasattr(target, "execute"):
        try:
            signature = inspect.signature(target)
            signature.bind()
        except (TypeError, ValueError):
            runtime = _FunctionRuntime(target)
        else:
            runtime = target()
    else:
        runtime = target
    if not callable(getattr(runtime, "execute", None)):
        raise TypeError(
            "worker entrypoint must be a zero-argument runtime factory, "
            "a runtime object, or a three-argument handler"
        )
    if not callable(getattr(runtime, "unload", None)):
        raise TypeError("worker runtime must provide unload()")
    return runtime


def _effective_policy(spec: WorkerSpec) -> SandboxPolicy:
    base = spec.sandbox or default_policy_for(spec.kind.value)
    if spec.sandbox_allow_subprocess is None:
        return base
    return SandboxPolicy(
        allow_network=base.allow_network,
        allow_subprocess=spec.sandbox_allow_subprocess,
        inherit_environment=base.inherit_environment,
        environment=base.environment,
        working_directory=base.working_directory,
        filesystem_read_roots=base.filesystem_read_roots,
        filesystem_write_roots=base.filesystem_write_roots,
        allow_runtime_imports=base.allow_runtime_imports,
        umask=base.umask,
        max_open_files=base.max_open_files,
        max_processes=base.max_processes,
        max_address_space_bytes=base.max_address_space_bytes,
    )


@dataclass(slots=True)
class _Execution:
    command: WorkerCommand
    cancellation: threading.Event
    thread: threading.Thread


def _send(connection: Connection, event: WorkerEvent) -> None:
    connection.send_bytes(encode_frame(event))


def _event(
    spec: WorkerSpec,
    kind: WorkerEventKind,
    *,
    command_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> WorkerEvent:
    return WorkerEvent(
        worker_id=spec.worker_id,
        kind=kind,
        timestamp=time.monotonic(),
        command_id=command_id,
        payload={} if payload is None else payload,
    )


def worker_main(connection: Connection, spec_frame: bytes) -> None:
    """Multiprocessing target. All heavy runtime imports happen below this boundary."""

    cwd_str = str(Path.cwd().resolve())
    if cwd_str not in sys.path:
        sys.path.insert(0, cwd_str)

    spec = decode_frame(spec_frame, WorkerSpec)
    if os.name == "posix":
        try:
            os.setsid()
        except PermissionError:
            pass
    try:
        apply_sandbox(
            _effective_policy(spec),
            environment_overrides=spec.thread_environment,
        )
        # Resolve third-party code only after the irreversible sandbox is active.
        runtime = _load_runtime(spec.entrypoint)
    except BaseException as exc:
        try:
            _send(
                connection,
                _event(
                    spec,
                    WorkerEventKind.ERROR,
                    payload={
                        "phase": "startup",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                ),
            )
        except BaseException:
            pass
        finally:
            connection.close()
        return
    results: queue.SimpleQueue[tuple[str, bool, Any]] = queue.SimpleQueue()
    active: _Execution | None = None

    def execute(command: WorkerCommand, cancellation: threading.Event) -> None:
        try:
            result = runtime.execute(
                command.operation or "", command.payload, cancellation
            )
        except CancelledExecution as exc:
            results.put((command.command_id, False, exc))
        except BaseException as exc:
            results.put((command.command_id, False, exc))
        else:
            results.put((command.command_id, True, result))

    try:
        _send(
            connection,
            _event(
                spec,
                WorkerEventKind.STARTED,
                payload={"pid": os.getpid(), "process_group": os.getpgrp()},
            ),
        )
        _send(connection, _event(spec, WorkerEventKind.READY))
        next_heartbeat = time.monotonic()
        shutting_down = False
        while not shutting_down:
            now = time.monotonic()
            if now >= next_heartbeat:
                _send(
                    connection,
                    _event(
                        spec,
                        WorkerEventKind.HEARTBEAT,
                        command_id=None
                        if active is None
                        else active.command.command_id,
                        payload={"busy": active is not None},
                    ),
                )
                next_heartbeat = now + spec.heartbeat_interval

            try:
                command_id, succeeded, outcome = results.get_nowait()
            except queue.Empty:
                pass
            else:
                if active is not None and active.command.command_id == command_id:
                    if succeeded:
                        _send(
                            connection,
                            _event(
                                spec,
                                WorkerEventKind.RESULT,
                                command_id=command_id,
                                payload={"result": outcome},
                            ),
                        )
                    elif isinstance(outcome, CancelledExecution):
                        _send(
                            connection,
                            _event(
                                spec,
                                WorkerEventKind.CANCELLED,
                                command_id=command_id,
                                payload={"message": str(outcome)},
                            ),
                        )
                    else:
                        _send(
                            connection,
                            _event(
                                spec,
                                WorkerEventKind.ERROR,
                                command_id=command_id,
                                payload={
                                    "error_type": type(outcome).__name__,
                                    "message": str(outcome),
                                    "traceback": "".join(
                                        traceback.format_exception(outcome)
                                    )[-8192:],
                                },
                            ),
                        )
                    active = None

            if connection.poll(0.01):
                command = decode_frame(
                    connection.recv_bytes(MAX_FRAME_BYTES + 4),
                    WorkerCommand,
                )
                if command.kind is WorkerCommandKind.EXECUTE:
                    if active is not None:
                        _send(
                            connection,
                            _event(
                                spec,
                                WorkerEventKind.ERROR,
                                command_id=command.command_id,
                                payload={
                                    "error_type": "WorkerBusyError",
                                    "message": "worker already has an active command",
                                },
                            ),
                        )
                    else:
                        cancellation = threading.Event()
                        thread = threading.Thread(
                            target=execute,
                            args=(command, cancellation),
                            name=f"comfyng-exec-{command.command_id[:8]}",
                            daemon=True,
                        )
                        active = _Execution(command, cancellation, thread)
                        thread.start()
                elif command.kind is WorkerCommandKind.CANCEL:
                    if (
                        active is not None
                        and active.command.command_id == command.target_command_id
                    ):
                        active.cancellation.set()
                elif command.kind is WorkerCommandKind.PING:
                    _send(
                        connection,
                        _event(
                            spec,
                            WorkerEventKind.HEARTBEAT,
                            command_id=command.command_id,
                            payload={"busy": active is not None, "pong": True},
                        ),
                    )
                elif command.kind is WorkerCommandKind.UNLOAD:
                    if active is not None:
                        _send(
                            connection,
                            _event(
                                spec,
                                WorkerEventKind.ERROR,
                                command_id=command.command_id,
                                payload={
                                    "error_type": "WorkerBusyError",
                                    "message": "cannot unload a busy worker",
                                },
                            ),
                        )
                    else:
                        payload = runtime.unload()
                        _send(
                            connection,
                            _event(
                                spec,
                                WorkerEventKind.UNLOADED,
                                command_id=command.command_id,
                                payload={"result": dict(payload)},
                            ),
                        )
                elif command.kind is WorkerCommandKind.SHUTDOWN:
                    _send(connection, _event(spec, WorkerEventKind.STOPPING))
                    if active is not None:
                        active.cancellation.set()
                    runtime.unload()
                    shutting_down = True
        _send(connection, _event(spec, WorkerEventKind.STOPPED))
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        try:
            runtime.unload()
        except BaseException:
            pass
        connection.close()
