from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, TypeVar

import msgspec

from .sandbox import SandboxPolicy


# IPC carries commands and handles, never tensors/images/model blobs.
MAX_FRAME_BYTES = 1024 * 1024
_HEADER_BYTES = 4


class FrameError(ValueError):
    """Raised when an IPC frame is malformed or has an unexpected type."""


class WorkerKind(StrEnum):
    GPU_MODEL = "gpu_model"
    GPU_AUX = "gpu_aux"
    CPU_COMPUTE = "cpu_compute"
    CPU_LIGHT = "cpu_light"
    IO = "io"
    DOWNLOAD = "download"
    METADATA = "metadata"
    PLUGIN = "plugin"
    ENCODER = "encoder"
    VAE = "vae"
    VIDEO = "video"


class WorkerCommandKind(StrEnum):
    EXECUTE = "execute"
    CANCEL = "cancel"
    PING = "ping"
    UNLOAD = "unload"
    SHUTDOWN = "shutdown"


class WorkerEventKind(StrEnum):
    STARTED = "started"
    READY = "ready"
    HEARTBEAT = "heartbeat"
    RESULT = "result"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNLOADED = "unloaded"
    STOPPING = "stopping"
    STOPPED = "stopped"


def _validate_identifier(value: str, field: str) -> None:
    if not value or len(value) > 160 or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed identifier")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must contain valid Unicode") from exc


class WorkerSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    worker_id: str
    kind: WorkerKind
    entrypoint: str | None = None
    start_method: str = "auto"
    heartbeat_interval: float = 1.0
    heartbeat_timeout: float = 5.0
    startup_timeout: float = 10.0
    shutdown_timeout: float = 2.0
    cancellation_grace: float = 0.25
    restart_limit: int = 3
    restart_window: float = 60.0
    sandbox: SandboxPolicy | None = None
    sandbox_allow_subprocess: bool | None = None
    thread_environment: Mapping[str, str] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.worker_id, "worker_id")
        if self.entrypoint is not None:
            _validate_identifier(self.entrypoint, "entrypoint")
            if ":" not in self.entrypoint:
                raise ValueError("entrypoint must use the 'module:attribute' form")
        if self.start_method not in {"auto", "spawn", "forkserver"}:
            raise ValueError("start_method must be auto, spawn or forkserver")
        if not math.isfinite(self.heartbeat_interval) or self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be a finite positive number")
        if (
            not math.isfinite(self.heartbeat_timeout)
            or self.heartbeat_timeout <= self.heartbeat_interval
        ):
            raise ValueError("heartbeat_timeout must exceed heartbeat_interval")
        for field, value in (
            ("startup_timeout", self.startup_timeout),
            ("shutdown_timeout", self.shutdown_timeout),
            ("cancellation_grace", self.cancellation_grace),
            ("restart_window", self.restart_window),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be a finite positive number")
        if self.restart_limit < 0:
            raise ValueError("restart_limit must be non-negative")
        for key, value in self.thread_environment.items():
            if not key or not isinstance(value, str):
                raise ValueError(
                    "thread_environment must contain string key/value pairs"
                )


class WorkerCommand(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    command_id: str
    kind: WorkerCommandKind
    operation: str | None = None
    payload: Mapping[str, Any] = msgspec.field(default_factory=dict)
    target_command_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.command_id, "command_id")
        if self.kind is WorkerCommandKind.EXECUTE:
            if self.operation is None:
                raise ValueError("execute commands require an operation")
            _validate_identifier(self.operation, "operation")
        elif self.operation is not None:
            raise ValueError("only execute commands may define an operation")
        if self.kind is WorkerCommandKind.CANCEL and not self.target_command_id:
            raise ValueError("cancel commands require target_command_id")


class WorkerEvent(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    worker_id: str
    kind: WorkerEventKind
    timestamp: float
    command_id: str | None = None
    payload: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.worker_id, "worker_id")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative number")


MessageT = TypeVar("MessageT", WorkerSpec, WorkerCommand, WorkerEvent)


def encode_frame(message: WorkerSpec | WorkerCommand | WorkerEvent) -> bytes:
    envelope = {
        "message_type": type(message).__name__,
        "payload": msgspec.to_builtins(message),
    }
    try:
        body = msgspec.msgpack.encode(envelope)
    except (TypeError, ValueError) as exc:
        raise FrameError(f"message cannot be encoded: {exc}") from exc
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError(f"frame exceeds {MAX_FRAME_BYTES} bytes")
    return len(body).to_bytes(_HEADER_BYTES, "big") + body


def decode_frame(
    frame: bytes | bytearray | memoryview, expected: type[MessageT]
) -> MessageT:
    value = bytes(frame)
    if len(value) < _HEADER_BYTES:
        raise FrameError("frame is missing its length header")
    declared = int.from_bytes(value[:_HEADER_BYTES], "big")
    body = value[_HEADER_BYTES:]
    if declared != len(body):
        raise FrameError(
            f"frame length mismatch: declared {declared}, received {len(body)}"
        )
    if declared > MAX_FRAME_BYTES:
        raise FrameError(f"frame exceeds {MAX_FRAME_BYTES} bytes")
    try:
        envelope = msgspec.msgpack.decode(body)
    except (TypeError, ValueError, msgspec.DecodeError) as exc:
        raise FrameError(f"frame payload is invalid: {exc}") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"message_type", "payload"}:
        raise FrameError("frame envelope must contain message_type and payload")
    if envelope["message_type"] != expected.__name__:
        raise FrameError(
            f"frame contains {envelope['message_type']!r}, expected {expected.__name__}"
        )
    try:
        return msgspec.convert(envelope["payload"], type=expected, strict=True)
    except (TypeError, ValueError, msgspec.ValidationError) as exc:
        raise FrameError(f"invalid {expected.__name__} frame: {exc}") from exc
