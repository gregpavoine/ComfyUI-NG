from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .heartbeat import HeartbeatWatchdog
from .process import (
    WorkerTransport,
    start_worker_process,
    terminate_process_tree,
)
from .protocol import (
    WorkerCommand,
    WorkerCommandKind,
    WorkerEvent,
    WorkerEventKind,
    WorkerSpec,
)
from .shared_memory import SharedObjectStore


class WorkerSupervisorError(RuntimeError):
    """Base error raised by supervised worker operations."""


class WorkerUnavailableError(WorkerSupervisorError):
    pass


class WorkerBusyError(WorkerSupervisorError):
    pass


class WorkerCrashedError(WorkerSupervisorError):
    def __init__(self, worker_id: str, reason: str) -> None:
        self.worker_id = worker_id
        self.reason = reason
        super().__init__(f"worker {worker_id!r} crashed: {reason}")


class WorkerTimeoutError(WorkerSupervisorError):
    def __init__(self, worker_id: str, timeout: float) -> None:
        self.worker_id = worker_id
        self.timeout = timeout
        super().__init__(
            f"worker {worker_id!r} exceeded execution timeout {timeout:.3f}s"
        )


class WorkerCancelledError(WorkerSupervisorError):
    pass


class CircuitOpenError(WorkerUnavailableError):
    pass


class RemoteExecutionError(WorkerSupervisorError):
    def __init__(
        self,
        worker_id: str,
        error_type: str,
        message: str,
        remote_traceback: str = "",
    ) -> None:
        self.worker_id = worker_id
        self.error_type = error_type
        self.remote_message = message
        self.remote_traceback = remote_traceback
        super().__init__(f"{worker_id}: {error_type}: {message}")


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    worker_id: str
    kind: str
    status: str
    pid: int
    start_method: str
    ready: bool
    busy: bool
    circuit_open: bool
    generation: int
    restart_count: int
    last_heartbeat: float
    last_error: str | None


@dataclass(slots=True)
class _WorkerRecord:
    spec: WorkerSpec
    condition: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.RLock())
    )
    transport: WorkerTransport | None = None
    monitor: threading.Thread | None = None
    watchdog: HeartbeatWatchdog | None = None
    status: str = "starting"
    ready: bool = False
    circuit_open: bool = False
    stop_requested: bool = False
    generation: int = 0
    restart_count: int = 0
    restart_times: deque[float] = field(default_factory=deque)
    pending: set[str] = field(default_factory=set)
    outcomes: dict[str, tuple[str, Any]] = field(default_factory=dict)
    force_restart_reason: str | None = None
    last_error: str | None = None


class WorkerSupervisor:
    """Owns worker lifecycles while keeping faults outside the API process."""

    def __init__(self, *, shared_store: SharedObjectStore | None = None) -> None:
        self.shared_store = shared_store or SharedObjectStore()
        self._owns_shared_store = shared_store is None
        self._records: dict[str, _WorkerRecord] = {}
        self._records_lock = threading.RLock()
        self._closed = False

    def __enter__(self) -> WorkerSupervisor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self, spec: WorkerSpec) -> WorkerSnapshot:
        with self._records_lock:
            if self._closed:
                raise WorkerUnavailableError("worker supervisor is closed")
            previous = self._records.get(spec.worker_id)
            if previous is not None and previous.status != "stopped":
                raise WorkerSupervisorError(f"worker {spec.worker_id!r} already exists")
            record = _WorkerRecord(spec=spec)
            self._records[spec.worker_id] = record
        try:
            self._spawn(record)
        except BaseException:
            with self._records_lock:
                self._records.pop(spec.worker_id, None)
            raise
        monitor = threading.Thread(
            target=self._monitor,
            args=(record,),
            name=f"comfyng-supervisor-{spec.worker_id}",
            daemon=True,
        )
        record.monitor = monitor
        monitor.start()
        return self.snapshot(spec.worker_id)

    def _spawn(self, record: _WorkerRecord) -> None:
        transport = start_worker_process(record.spec)
        with record.condition:
            record.transport = transport
            record.generation += 1
            record.status = "starting"
            record.ready = False
            record.watchdog = HeartbeatWatchdog(record.spec.heartbeat_timeout)
            record.condition.notify_all()
        deadline = time.monotonic() + record.spec.startup_timeout
        try:
            while time.monotonic() < deadline:
                if transport.poll(min(0.05, max(0.0, deadline - time.monotonic()))):
                    try:
                        event = transport.receive()
                    except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
                        raise WorkerCrashedError(
                            record.spec.worker_id,
                            f"startup IPC closed before ready: {exc}",
                        ) from exc
                    if (
                        event.kind is WorkerEventKind.ERROR
                        and event.command_id is None
                        and event.payload.get("phase") == "startup"
                    ):
                        raise WorkerCrashedError(
                            record.spec.worker_id,
                            "startup failed: "
                            f"{event.payload.get('error_type', 'Error')}: "
                            f"{event.payload.get('message', '')}",
                        )
                    self._consume_event(record, event)
                    if event.kind is WorkerEventKind.READY:
                        return
                if not transport.is_alive():
                    raise WorkerCrashedError(
                        record.spec.worker_id,
                        f"exited during startup with code {transport.exitcode}",
                    )
            raise WorkerTimeoutError(record.spec.worker_id, record.spec.startup_timeout)
        except BaseException:
            terminate_process_tree(transport, grace=record.spec.shutdown_timeout)
            try:
                transport.close()
            except (OSError, ValueError):
                pass
            raise

    def _monitor(self, record: _WorkerRecord) -> None:
        while True:
            with record.condition:
                if record.stop_requested:
                    return
                transport = record.transport
                watchdog = record.watchdog
                forced = record.force_restart_reason
                record.force_restart_reason = None
            if transport is None:
                return
            if forced is not None:
                self._recover(record, transport, forced)
                continue
            try:
                if transport.poll(0.02):
                    self._consume_event(record, transport.receive())
            except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
                self._recover(record, transport, f"IPC failure: {exc}")
                continue
            if not transport.is_alive():
                self._recover(
                    record,
                    transport,
                    f"exited with code {transport.exitcode}",
                )
                continue
            with record.condition:
                is_ready = record.ready and record.transport is transport
            if is_ready and watchdog is not None and watchdog.expired():
                self._recover(record, transport, "heartbeat timeout")

    def _consume_event(self, record: _WorkerRecord, event: WorkerEvent) -> None:
        if event.worker_id != record.spec.worker_id:
            raise ValueError(f"worker event identity mismatch: {event.worker_id!r}")
        with record.condition:
            if record.watchdog is not None:
                record.watchdog.observe()
            if event.kind is WorkerEventKind.STARTED:
                transport = record.transport
                if transport is not None:
                    pid = event.payload.get("pid")
                    process_group = event.payload.get("process_group")
                    if (
                        not isinstance(pid, int)
                        or isinstance(pid, bool)
                        or pid != transport.pid
                    ):
                        raise ValueError("worker reported an invalid PID")
                    # Only a dedicated session/process group is safe to signal. Never
                    # retain an inherited group that could contain the API process.
                    if (
                        isinstance(process_group, int)
                        and not isinstance(process_group, bool)
                        and process_group == transport.pid
                    ):
                        transport.process_group = process_group
            elif event.kind is WorkerEventKind.READY:
                record.status = "ready"
                record.ready = True
            elif event.kind is WorkerEventKind.HEARTBEAT:
                if record.ready:
                    record.status = (
                        "busy" if bool(event.payload.get("busy")) else "ready"
                    )
            elif event.kind in {WorkerEventKind.RESULT, WorkerEventKind.UNLOADED}:
                if event.command_id is not None:
                    record.outcomes[event.command_id] = (
                        "result",
                        event.payload.get("result"),
                    )
                    record.pending.discard(event.command_id)
                    if record.ready:
                        record.status = "ready"
            elif event.kind is WorkerEventKind.ERROR:
                if event.command_id is not None:
                    record.outcomes[event.command_id] = (
                        "error",
                        RemoteExecutionError(
                            record.spec.worker_id,
                            str(event.payload.get("error_type", "RemoteError")),
                            str(
                                event.payload.get("message", "worker execution failed")
                            ),
                            str(event.payload.get("traceback", "")),
                        ),
                    )
                    record.pending.discard(event.command_id)
                    if record.ready:
                        record.status = "ready"
            elif event.kind is WorkerEventKind.CANCELLED:
                if event.command_id is not None:
                    record.outcomes[event.command_id] = (
                        "cancelled",
                        WorkerCancelledError(
                            f"worker {record.spec.worker_id!r} cancelled command "
                            f"{event.command_id!r}"
                        ),
                    )
                    record.pending.discard(event.command_id)
                    if record.ready:
                        record.status = "ready"
            elif event.kind is WorkerEventKind.STOPPING:
                record.status = "stopping"
                record.ready = False
            elif event.kind is WorkerEventKind.STOPPED:
                record.status = "stopped"
                record.ready = False
            record.condition.notify_all()

    def _recover(
        self,
        record: _WorkerRecord,
        transport: WorkerTransport,
        reason: str,
    ) -> None:
        with record.condition:
            if record.transport is not transport or record.stop_requested:
                return
            record.status = "failed"
            record.ready = False
            record.last_error = reason
        terminate_process_tree(transport, grace=record.spec.shutdown_timeout)
        try:
            transport.close()
        except (OSError, ValueError):
            pass
        self.shared_store.cleanup_owner(record.spec.worker_id)
        with record.condition:
            failure = WorkerCrashedError(record.spec.worker_id, reason)
            for command_id in tuple(record.pending):
                record.outcomes[command_id] = ("error", failure)
            record.pending.clear()
            now = time.monotonic()
            while (
                record.restart_times
                and now - record.restart_times[0] > record.spec.restart_window
            ):
                record.restart_times.popleft()
            if len(record.restart_times) >= record.spec.restart_limit:
                record.circuit_open = True
                record.status = "circuit_open"
                record.transport = None
                record.condition.notify_all()
                return
            record.restart_times.append(now)
            record.restart_count += 1
            record.status = "restarting"
            record.transport = None
            record.condition.notify_all()
        try:
            self._spawn(record)
        except BaseException as exc:
            with record.condition:
                record.circuit_open = True
                record.status = "circuit_open"
                record.last_error = f"restart failed: {exc}"
                record.transport = None
                record.condition.notify_all()

    def snapshot(self, worker_id: str) -> WorkerSnapshot:
        record = self._get_record(worker_id)
        with record.condition:
            transport = record.transport
            watchdog = record.watchdog
            return WorkerSnapshot(
                worker_id=record.spec.worker_id,
                kind=record.spec.kind.value,
                status=record.status,
                pid=0 if transport is None else transport.pid,
                start_method="" if transport is None else transport.start_method,
                ready=record.ready,
                busy=bool(record.pending),
                circuit_open=record.circuit_open,
                generation=record.generation,
                restart_count=record.restart_count,
                last_heartbeat=0.0 if watchdog is None else watchdog.last_seen,
                last_error=record.last_error,
            )

    def execute(
        self,
        worker_id: str,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if not operation:
            raise ValueError("operation must not be empty")
        command = WorkerCommand(
            command_id=uuid4().hex,
            kind=WorkerCommandKind.EXECUTE,
            operation=operation,
            payload={} if payload is None else dict(payload),
        )
        return self._request(worker_id, command, timeout=timeout)

    def unload(self, worker_id: str, *, timeout: float | None = None) -> Any:
        command = WorkerCommand(
            command_id=uuid4().hex,
            kind=WorkerCommandKind.UNLOAD,
        )
        return self._request(worker_id, command, timeout=timeout)

    def cancel(self, worker_id: str) -> bool:
        """Request cooperative cancellation of the worker's active command."""

        record = self._get_record(worker_id)
        with record.condition:
            if not record.pending:
                return False
            if not record.ready or record.transport is None:
                raise WorkerUnavailableError(f"worker {worker_id!r} is {record.status}")
            target = next(iter(record.pending))
            command = WorkerCommand(
                command_id=uuid4().hex,
                kind=WorkerCommandKind.CANCEL,
                target_command_id=target,
            )
            try:
                record.transport.send(command)
            except (BrokenPipeError, EOFError, OSError) as exc:
                record.force_restart_reason = f"IPC cancellation failed: {exc}"
                record.condition.notify_all()
                raise WorkerCrashedError(worker_id, str(exc)) from exc
            return True

    def _request(
        self,
        worker_id: str,
        command: WorkerCommand,
        *,
        timeout: float | None,
    ) -> Any:
        record = self._get_record(worker_id)
        operation_timeout = timeout if timeout is not None else 30.0
        if operation_timeout <= 0:
            raise ValueError("timeout must be positive")
        startup_deadline = time.monotonic() + record.spec.startup_timeout
        with record.condition:
            while not record.ready and record.status in {"starting", "restarting"}:
                remaining = startup_deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerUnavailableError(
                        f"worker {worker_id!r} did not become ready"
                    )
                record.condition.wait(min(0.05, remaining))
            if record.circuit_open:
                raise CircuitOpenError(f"worker {worker_id!r} circuit is open")
            if not record.ready or record.transport is None:
                raise WorkerUnavailableError(f"worker {worker_id!r} is {record.status}")
            if record.pending:
                raise WorkerBusyError(f"worker {worker_id!r} is busy")
            transport = record.transport
            record.pending.add(command.command_id)
            record.status = "busy"
            try:
                transport.send(command)
            except (BrokenPipeError, EOFError, OSError) as exc:
                record.pending.discard(command.command_id)
                record.force_restart_reason = f"IPC send failed: {exc}"
                record.condition.notify_all()
                raise WorkerCrashedError(worker_id, str(exc)) from exc

            deadline = time.monotonic() + operation_timeout
            while command.command_id not in record.outcomes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                record.condition.wait(min(0.05, remaining))
            if command.command_id in record.outcomes:
                outcome_kind, outcome = record.outcomes.pop(command.command_id)
                if outcome_kind == "result":
                    return outcome
                raise outcome

            cancel = WorkerCommand(
                command_id=uuid4().hex,
                kind=WorkerCommandKind.CANCEL,
                target_command_id=command.command_id,
            )
            try:
                transport.send(cancel)
            except (BrokenPipeError, EOFError, OSError):
                pass
            cancel_deadline = time.monotonic() + record.spec.cancellation_grace
            while (
                command.command_id not in record.outcomes
                and time.monotonic() < cancel_deadline
            ):
                record.condition.wait(
                    min(0.05, max(0.0, cancel_deadline - time.monotonic()))
                )
            record.outcomes.pop(command.command_id, None)
            record.pending.discard(command.command_id)
            record.force_restart_reason = (
                f"execution timeout after {operation_timeout:.3f}s"
            )
            record.condition.notify_all()
        raise WorkerTimeoutError(worker_id, operation_timeout)

    def stop(self, worker_id: str) -> None:
        record = self._get_record(worker_id)
        with record.condition:
            if record.status == "stopped":
                return
            record.stop_requested = True
            record.ready = False
            record.status = "stopping"
            transport = record.transport
            if transport is not None and transport.is_alive():
                try:
                    transport.send(
                        WorkerCommand(
                            command_id=uuid4().hex,
                            kind=WorkerCommandKind.SHUTDOWN,
                        )
                    )
                except (BrokenPipeError, EOFError, OSError):
                    pass
            record.condition.notify_all()
        if transport is not None:
            transport.process.join(timeout=record.spec.shutdown_timeout)
            terminate_process_tree(transport, grace=record.spec.shutdown_timeout)
            try:
                transport.close()
            except (OSError, ValueError):
                pass
        monitor = record.monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=record.spec.shutdown_timeout)
        self.shared_store.cleanup_owner(worker_id)
        with record.condition:
            failure = WorkerUnavailableError(f"worker {worker_id!r} was stopped")
            for command_id in tuple(record.pending):
                record.outcomes[command_id] = ("error", failure)
            record.pending.clear()
            record.transport = None
            record.status = "stopped"
            record.condition.notify_all()

    def close(self) -> None:
        with self._records_lock:
            if self._closed:
                return
            worker_ids = tuple(self._records)
            self._closed = True
        for worker_id in worker_ids:
            self.stop(worker_id)
        if self._owns_shared_store:
            self.shared_store.close()

    def _get_record(self, worker_id: str) -> _WorkerRecord:
        with self._records_lock:
            record = self._records.get(worker_id)
        if record is None:
            raise WorkerUnavailableError(f"unknown worker {worker_id!r}")
        return record
