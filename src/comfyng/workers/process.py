from __future__ import annotations

import multiprocessing
import os
import signal
import sys
from dataclasses import dataclass
from multiprocessing.connection import Connection

from comfyng.runtime.entrypoint import worker_main

from .protocol import (
    MAX_FRAME_BYTES,
    WorkerCommand,
    WorkerEvent,
    WorkerKind,
    WorkerSpec,
    decode_frame,
    encode_frame,
)


_CUDA_WORKERS = frozenset(
    {
        WorkerKind.GPU_MODEL,
        WorkerKind.GPU_AUX,
        WorkerKind.ENCODER,
        WorkerKind.VAE,
        WorkerKind.VIDEO,
    }
)


def select_start_method(
    kind: WorkerKind,
    *,
    requested: str = "auto",
    platform_name: str = sys.platform,
    available_methods: tuple[str, ...] | list[str] | None = None,
) -> str:
    available = tuple(available_methods or multiprocessing.get_all_start_methods())
    if requested == "fork":
        raise ValueError("fork is forbidden for ComfyUI-NG workers")
    if requested not in {"auto", "spawn", "forkserver"}:
        raise ValueError(f"unsupported multiprocessing start method: {requested}")
    if platform_name == "darwin" or kind in _CUDA_WORKERS:
        selected = "spawn"
    elif requested == "auto":
        selected = "forkserver" if "forkserver" in available else "spawn"
    else:
        selected = requested
    if selected == "forkserver" and "forkserver" not in available:
        selected = "spawn"
    if selected not in available:
        raise RuntimeError(f"multiprocessing start method {selected!r} is unavailable")
    return selected


@dataclass(slots=True)
class WorkerTransport:
    spec: WorkerSpec
    start_method: str
    process: multiprocessing.Process
    connection: Connection
    process_group: int | None = None

    @property
    def pid(self) -> int:
        return self.process.pid or 0

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def send(self, command: WorkerCommand) -> None:
        self.connection.send_bytes(encode_frame(command))

    def poll(self, timeout: float = 0.0) -> bool:
        return self.connection.poll(timeout)

    def receive(self) -> WorkerEvent:
        return decode_frame(
            self.connection.recv_bytes(MAX_FRAME_BYTES + 4),
            WorkerEvent,
        )

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            self.process.close()


def start_worker_process(spec: WorkerSpec) -> WorkerTransport:
    start_method = select_start_method(
        spec.kind,
        requested=spec.start_method,
    )
    context = multiprocessing.get_context(start_method)
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=worker_main,
        args=(child_connection, encode_frame(spec)),
        name=f"comfyng-{spec.kind.value}-{spec.worker_id}",
        daemon=False,
    )
    try:
        process.start()
    except BaseException:
        parent_connection.close()
        child_connection.close()
        raise
    child_connection.close()
    return WorkerTransport(spec, start_method, process, parent_connection)


def terminate_process_tree(
    transport: WorkerTransport,
    *,
    grace: float,
) -> None:
    """Terminate the worker's POSIX process group, then hard-kill survivors."""

    process = transport.process
    if process.pid is None:
        return
    pgid = transport.process_group
    group_signalled = False
    if os.name == "posix" and pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
            group_signalled = True
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.is_alive():
                process.terminate()
    elif process.is_alive():
        process.terminate()
    process.join(timeout=max(0.0, grace))
    if process.is_alive():
        if os.name == "posix" and pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
                group_signalled = True
            except ProcessLookupError:
                pass
            except PermissionError:
                process.kill()
        else:
            process.kill()
        process.join(timeout=max(0.1, grace))
    elif os.name == "posix" and pgid is not None and group_signalled:
        # The leader may have exited while descendants still own the process group.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        transport.connection.close()
    except OSError:
        pass
