from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import json
import math
from typing import Any, Protocol

from comfyng.core.json_values import FrozenDict, freeze_json_value
from comfyng.database.connection import Database
from comfyng.resources.budgets import (
    FallbackAction,
    ResourceAlternative,
    ResourceEstimate,
)


_QUEUE_NAMES = frozenset(
    {"interactive", "normal", "batch", "background", "download", "maintenance"}
)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {JobStatus.PREPARING, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.PREPARING: frozenset(
        {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.RUNNING: frozenset(
        {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}
_STATUS_POSITION = {
    JobStatus.QUEUED: 0,
    JobStatus.PREPARING: 1,
    JobStatus.RUNNING: 2,
    JobStatus.COMPLETED: 3,
    JobStatus.FAILED: 3,
    JobStatus.CANCELLED: 3,
}


class JobError(RuntimeError):
    pass


class JobNotFound(JobError, KeyError):
    pass


class DuplicateJob(JobError):
    pass


class InvalidJobTransition(JobError, ValueError):
    def __init__(self, source: JobStatus, target: JobStatus, reason: str = "") -> None:
        self.source = source
        self.target = target
        self.reason = reason
        suffix = f": {reason}" if reason else ""
        super().__init__(
            f"invalid job transition {source.value} -> {target.value}{suffix}"
        )


class JobTransitionConflict(JobError):
    def __init__(self, job_id: str, expected: JobStatus, actual: JobStatus) -> None:
        self.job_id = job_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"job {job_id!r} expected {expected.value}, found {actual.value}"
        )


def _finite_non_negative(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 160
    ):
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc
    return value


@dataclass(frozen=True, slots=True)
class JobSubmission:
    job_id: str
    queue: str = "normal"
    user_priority: int = 50
    payload: Mapping[str, Any] = field(default_factory=FrozenDict)
    max_attempts: int = 3
    workflow_id: str | None = None
    workflow_version_id: int | None = None
    cache_key: str | None = None
    resource_estimate: ResourceEstimate | None = None
    warm_model_bonus: float = 0.0
    cache_reuse_bonus: float = 0.0
    memory_pressure_penalty: float = 0.0
    estimated_duration_penalty: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.job_id, "job_id")
        if not isinstance(self.queue, str) or self.queue not in _QUEUE_NAMES:
            raise ValueError(f"unknown scheduler queue: {self.queue!r}")
        if (
            isinstance(self.user_priority, bool)
            or not isinstance(self.user_priority, int)
            or not 0 <= self.user_priority <= 100
        ):
            raise ValueError("user_priority must be an integer between 0 and 100")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if (self.workflow_id is None) != (self.workflow_version_id is None):
            raise ValueError(
                "workflow_id and workflow_version_id must be specified together"
            )
        if self.workflow_id is not None:
            _identifier(self.workflow_id, "workflow_id")
        if self.workflow_version_id is not None and (
            isinstance(self.workflow_version_id, bool)
            or not isinstance(self.workflow_version_id, int)
            or self.workflow_version_id < 1
        ):
            raise ValueError("workflow_version_id must be a positive integer or None")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a JSON object")
        object.__setattr__(
            self, "payload", freeze_json_value(self.payload, path="$.payload")
        )
        if self.cache_key is not None:
            _identifier(self.cache_key, "cache_key")
        if self.resource_estimate is not None and not isinstance(
            self.resource_estimate, ResourceEstimate
        ):
            raise ValueError("resource_estimate must be ResourceEstimate or None")
        for name in (
            "warm_model_bonus",
            "cache_reuse_bonus",
            "memory_pressure_penalty",
            "estimated_duration_penalty",
        ):
            _finite_non_negative(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    queue: str
    user_priority: int
    payload: Mapping[str, Any]
    max_attempts: int
    workflow_id: str | None
    workflow_version_id: int | None
    cache_key: str | None
    resource_estimate: ResourceEstimate | None
    warm_model_bonus: float
    cache_reuse_bonus: float
    memory_pressure_penalty: float
    estimated_duration_penalty: float
    status: JobStatus
    attempt: int
    revision: int
    created_at: float
    queued_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _identifier(self.job_id, "job_id")
        if not isinstance(self.queue, str) or self.queue not in _QUEUE_NAMES:
            raise ValueError(f"unknown scheduler queue: {self.queue!r}")
        if (
            isinstance(self.user_priority, bool)
            or not isinstance(self.user_priority, int)
            or not 0 <= self.user_priority <= 100
        ):
            raise ValueError("user_priority must be an integer between 0 and 100")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if (self.workflow_id is None) != (self.workflow_version_id is None):
            raise ValueError(
                "workflow_id and workflow_version_id must be specified together"
            )
        if self.workflow_id is not None:
            _identifier(self.workflow_id, "workflow_id")
        if self.workflow_version_id is not None and (
            isinstance(self.workflow_version_id, bool)
            or not isinstance(self.workflow_version_id, int)
            or self.workflow_version_id < 1
        ):
            raise ValueError("workflow_version_id must be a positive integer or None")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a JSON object")
        if self.cache_key is not None:
            _identifier(self.cache_key, "cache_key")
        if self.resource_estimate is not None and not isinstance(
            self.resource_estimate, ResourceEstimate
        ):
            raise ValueError("resource_estimate must be ResourceEstimate or None")
        for name in (
            "warm_model_bonus",
            "cache_reuse_bonus",
            "memory_pressure_penalty",
            "estimated_duration_penalty",
        ):
            _finite_non_negative(name, getattr(self, name))
        if not isinstance(self.status, JobStatus):
            raise ValueError("status must be JobStatus")
        for name in ("attempt", "revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.attempt >= self.max_attempts:
            raise ValueError("attempt must be lower than max_attempts")
        for name in ("created_at", "queued_at", "updated_at"):
            _finite_non_negative(name, getattr(self, name))
        if self.started_at is not None:
            _finite_non_negative("started_at", self.started_at)
        if self.finished_at is not None:
            _finite_non_negative("finished_at", self.finished_at)
        if not self.created_at <= self.queued_at <= self.updated_at:
            raise ValueError("job timestamps must be monotonic")
        if (
            self.started_at is not None
            and not self.created_at <= self.started_at <= self.updated_at
        ):
            raise ValueError("started_at must fall within the job lifetime")
        if self.finished_at is not None and self.finished_at != self.updated_at:
            raise ValueError("finished_at must equal updated_at for terminal revisions")
        if self.status is JobStatus.RUNNING and self.started_at is None:
            raise ValueError("running jobs require started_at")
        if (
            self.status in {JobStatus.QUEUED, JobStatus.PREPARING}
            and self.started_at is not None
        ):
            raise ValueError("queued/preparing jobs cannot have started_at")
        if self.status.terminal != (self.finished_at is not None):
            raise ValueError("finished_at presence must match terminal status")
        if self.status is not JobStatus.COMPLETED and self.result is not None:
            raise ValueError("only completed jobs may carry a result")
        object.__setattr__(
            self, "payload", freeze_json_value(self.payload, path="$.payload")
        )
        if self.result is not None:
            object.__setattr__(
                self, "result", freeze_json_value(self.result, path="$.result")
            )
        if self.error is not None:
            frozen = freeze_json_value(self.error, path="$.error")
            if not isinstance(frozen, FrozenDict):
                raise ValueError("error must be a JSON object")
            object.__setattr__(self, "error", frozen)

    @property
    def monotonic_position(self) -> int:
        """Lifecycle position that remains increasing across retry attempts."""

        return self.attempt * 4 + _STATUS_POSITION[self.status]

    @classmethod
    def from_submission(cls, submission: JobSubmission, *, now: float) -> JobRecord:
        timestamp = _finite_non_negative("now", now)
        return cls(
            job_id=submission.job_id,
            queue=submission.queue,
            user_priority=submission.user_priority,
            payload=submission.payload,
            max_attempts=submission.max_attempts,
            workflow_id=submission.workflow_id,
            workflow_version_id=submission.workflow_version_id,
            cache_key=submission.cache_key,
            resource_estimate=submission.resource_estimate,
            warm_model_bonus=submission.warm_model_bonus,
            cache_reuse_bonus=submission.cache_reuse_bonus,
            memory_pressure_penalty=submission.memory_pressure_penalty,
            estimated_duration_penalty=submission.estimated_duration_penalty,
            status=JobStatus.QUEUED,
            attempt=0,
            revision=0,
            created_at=timestamp,
            queued_at=timestamp,
            updated_at=timestamp,
        )


class JobRepository(Protocol):
    async def create(self, submission: JobSubmission, *, now: float) -> JobRecord: ...
    async def get(self, job_id: str) -> JobRecord | None: ...
    async def list(self) -> tuple[JobRecord, ...]: ...
    async def transition(
        self,
        job_id: str,
        *,
        expected: JobStatus,
        target: JobStatus,
        now: float,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
    ) -> JobRecord: ...
    async def retry(
        self,
        job_id: str,
        *,
        now: float,
        expected: JobStatus = JobStatus.FAILED,
        error: Mapping[str, Any] | None = None,
    ) -> JobRecord: ...


def _transition_record(
    current: JobRecord,
    *,
    expected: JobStatus,
    target: JobStatus,
    now: float,
    result: Any = None,
    error: Mapping[str, Any] | None = None,
) -> JobRecord:
    if current.status is not expected:
        raise JobTransitionConflict(current.job_id, expected, current.status)
    if target not in _TRANSITIONS[current.status]:
        raise InvalidJobTransition(current.status, target)
    if now < current.updated_at:
        raise ValueError("transition time cannot move backwards")
    if result is not None and target is not JobStatus.COMPLETED:
        raise ValueError("result is only valid for completed jobs")
    if error is not None and target not in {JobStatus.FAILED, JobStatus.CANCELLED}:
        raise ValueError("error is only valid for failed or cancelled jobs")
    return replace(
        current,
        status=target,
        revision=current.revision + 1,
        updated_at=now,
        started_at=(now if target is JobStatus.RUNNING else current.started_at),
        finished_at=(now if target.terminal else None),
        result=result,
        error=error,
    )


def _retry_record(
    current: JobRecord,
    *,
    expected: JobStatus,
    now: float,
    error: Mapping[str, Any] | None = None,
) -> JobRecord:
    if current.status is not expected:
        raise JobTransitionConflict(current.job_id, expected, current.status)
    if expected not in {JobStatus.FAILED, JobStatus.PREPARING, JobStatus.RUNNING}:
        raise InvalidJobTransition(expected, JobStatus.QUEUED, "not retryable")
    if current.attempt + 1 >= current.max_attempts:
        raise InvalidJobTransition(expected, JobStatus.QUEUED, "retry budget exhausted")
    if now < current.updated_at:
        raise ValueError("retry time cannot move backwards")
    return replace(
        current,
        status=JobStatus.QUEUED,
        attempt=current.attempt + 1,
        revision=current.revision + 1,
        queued_at=now,
        updated_at=now,
        started_at=None,
        finished_at=None,
        result=None,
        error=error,
    )


class InMemoryJobRepository:
    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}
        self._history: dict[str, list[JobRecord]] = {}
        self._lock = asyncio.Lock()

    async def create(self, submission: JobSubmission, *, now: float) -> JobRecord:
        if not isinstance(submission, JobSubmission):
            raise ValueError("submission must be JobSubmission")
        async with self._lock:
            if submission.job_id in self._records:
                raise DuplicateJob(f"job {submission.job_id!r} already exists")
            record = JobRecord.from_submission(submission, now=now)
            self._records[record.job_id] = record
            self._history[record.job_id] = [record]
            return record

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._records.get(job_id)

    async def list(self) -> tuple[JobRecord, ...]:
        async with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda item: (item.created_at, item.job_id),
                )
            )

    async def transition(
        self,
        job_id: str,
        *,
        expected: JobStatus,
        target: JobStatus,
        now: float,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        if not isinstance(expected, JobStatus) or not isinstance(target, JobStatus):
            raise ValueError("expected and target must be JobStatus values")
        timestamp = _finite_non_negative("now", now)
        async with self._lock:
            current = self._records.get(job_id)
            if current is None:
                raise JobNotFound(job_id)
            updated = _transition_record(
                current,
                expected=expected,
                target=target,
                now=timestamp,
                result=result,
                error=error,
            )
            self._records[job_id] = updated
            self._history[job_id].append(updated)
            return updated

    async def retry(
        self,
        job_id: str,
        *,
        now: float,
        expected: JobStatus = JobStatus.FAILED,
        error: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        if not isinstance(expected, JobStatus):
            raise ValueError("expected must be JobStatus")
        timestamp = _finite_non_negative("now", now)
        async with self._lock:
            current = self._records.get(job_id)
            if current is None:
                raise JobNotFound(job_id)
            updated = _retry_record(
                current,
                expected=expected,
                now=timestamp,
                error=error,
            )
            self._records[job_id] = updated
            self._history[job_id].append(updated)
            return updated

    async def history(self, job_id: str) -> tuple[JobRecord, ...]:
        async with self._lock:
            if job_id not in self._history:
                raise JobNotFound(job_id)
            return tuple(self._history[job_id])


def _estimate_payload(estimate: ResourceEstimate | None) -> Any:
    if estimate is None:
        return None
    return {
        "cpu_cores": estimate.cpu_cores,
        "ram_bytes": estimate.ram_bytes,
        "vram_mb": estimate.vram_mb,
        "gpu_index": estimate.gpu_index,
        "heavy_gpu": estimate.heavy_gpu,
        "pinned_ram_bytes": estimate.pinned_ram_bytes,
        "concurrent_reads": estimate.concurrent_reads,
        "concurrent_writes": estimate.concurrent_writes,
        "alternatives": [
            {
                "action": alternative.action.value,
                "estimate": _estimate_payload(alternative.estimate),
            }
            for alternative in estimate.alternatives
        ],
    }


def _estimate_from_payload(value: Any) -> ResourceEstimate | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("stored resource estimate must be an object")
    alternatives: list[ResourceAlternative] = []
    raw_alternatives = value.get("alternatives", [])
    if not isinstance(raw_alternatives, list):
        raise ValueError("stored resource alternatives must be an array")
    for raw in raw_alternatives:
        if not isinstance(raw, Mapping):
            raise ValueError("stored resource alternative must be an object")
        nested = _estimate_from_payload(raw.get("estimate"))
        if nested is None:
            raise ValueError("stored resource alternative requires an estimate")
        alternatives.append(
            ResourceAlternative(FallbackAction(str(raw.get("action"))), nested)
        )
    return ResourceEstimate(
        cpu_cores=int(value["cpu_cores"]),
        ram_bytes=int(value["ram_bytes"]),
        vram_mb=int(value.get("vram_mb", 0)),
        gpu_index=(None if value.get("gpu_index") is None else int(value["gpu_index"])),
        heavy_gpu=bool(value.get("heavy_gpu", False)),
        pinned_ram_bytes=int(value.get("pinned_ram_bytes", 0)),
        concurrent_reads=int(value.get("concurrent_reads", 0)),
        concurrent_writes=int(value.get("concurrent_writes", 0)),
        alternatives=tuple(alternatives),
    )


def _iso_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _execution_payload(record: JobRecord) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queue": record.queue,
        "cache_key": record.cache_key,
        "resource_estimate": _estimate_payload(record.resource_estimate),
        "warm_model_bonus": record.warm_model_bonus,
        "cache_reuse_bonus": record.cache_reuse_bonus,
        "memory_pressure_penalty": record.memory_pressure_penalty,
        "estimated_duration_penalty": record.estimated_duration_penalty,
        "revision": record.revision,
        "result": record.result,
    }


def _row_record(row: Mapping[str, Any]) -> JobRecord:
    execution = json.loads(row["execution_json"])
    if not isinstance(execution, dict):
        raise ValueError("stored execution_json must be an object")
    payload = json.loads(row["inputs_json"])
    error = None if row["error_json"] is None else json.loads(row["error_json"])
    created_at = _parse_timestamp(str(row["created_at"]))
    queued_at = _parse_timestamp(str(row["queued_at"]))
    updated_at = _parse_timestamp(str(row["updated_at"]))
    if created_at is None or queued_at is None or updated_at is None:
        raise ValueError("stored job timestamps cannot be null")
    return JobRecord(
        job_id=str(row["id"]),
        queue=str(execution.get("queue", "normal")),
        user_priority=int(row["priority"]),
        payload=payload,
        max_attempts=int(row["max_attempts"]),
        workflow_id=str(row["workflow_id"]),
        workflow_version_id=int(row["workflow_version_id"]),
        cache_key=(
            None if execution.get("cache_key") is None else str(execution["cache_key"])
        ),
        resource_estimate=_estimate_from_payload(execution.get("resource_estimate")),
        warm_model_bonus=float(execution.get("warm_model_bonus", 0)),
        cache_reuse_bonus=float(execution.get("cache_reuse_bonus", 0)),
        memory_pressure_penalty=float(execution.get("memory_pressure_penalty", 0)),
        estimated_duration_penalty=float(
            execution.get("estimated_duration_penalty", 0)
        ),
        status=JobStatus(str(row["status"])),
        attempt=int(row["attempt"]),
        revision=int(execution.get("revision", 0)),
        created_at=created_at,
        queued_at=queued_at,
        updated_at=updated_at,
        started_at=_parse_timestamp(row["started_at"]),
        finished_at=_parse_timestamp(row["finished_at"]),
        result=execution.get("result"),
        error=error,
    )


class SqliteJobRepository:
    """Atomic durable JobRepository adapter over the V1 SQLite jobs table."""

    def __init__(self, database: Database) -> None:
        if not isinstance(database, Database):
            raise ValueError("database must be Database")
        self.database = database

    async def create(self, submission: JobSubmission, *, now: float) -> JobRecord:
        if not isinstance(submission, JobSubmission):
            raise ValueError("submission must be JobSubmission")
        if submission.workflow_id is None or submission.workflow_version_id is None:
            raise ValueError("SQLite jobs require workflow_id and workflow_version_id")
        record = JobRecord.from_submission(submission, now=now)
        timestamp = _iso_timestamp(record.created_at)
        try:
            row = await self.database.repositories.jobs.create(
                {
                    "id": record.job_id,
                    "workflow_id": record.workflow_id,
                    "workflow_version_id": record.workflow_version_id,
                    "status": record.status.value,
                    "inputs_json": record.payload,
                    "execution_json": _execution_payload(record),
                    "priority": record.user_priority,
                    "attempt": record.attempt,
                    "max_attempts": record.max_attempts,
                    "created_at": timestamp,
                    "queued_at": timestamp,
                    "updated_at": timestamp,
                }
            )
        except Exception as exc:
            if "UNIQUE constraint failed: jobs.id" in str(exc):
                raise DuplicateJob(f"job {record.job_id!r} already exists") from exc
            raise
        return _row_record(row)

    async def get(self, job_id: str) -> JobRecord | None:
        row = await self.database.repositories.jobs.get(job_id)
        return None if row is None else _row_record(row)

    async def list(self) -> tuple[JobRecord, ...]:
        rows = await self.database.repositories.jobs.list(
            order_by="created_at",
            limit=100_000,
        )
        return tuple(_row_record(row) for row in rows)

    @staticmethod
    async def _load_for_update(connection: Any, job_id: str) -> JobRecord:
        row = await (
            await connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return _row_record(dict(row))

    @staticmethod
    async def _persist(connection: Any, record: JobRecord) -> None:
        await connection.execute(
            "UPDATE jobs SET status = ?, execution_json = ?, attempt = ?, "
            "error_json = ?, queued_at = ?, started_at = ?, finished_at = ?, "
            "updated_at = ? WHERE id = ?",
            (
                record.status.value,
                _json(_execution_payload(record)),
                record.attempt,
                None if record.error is None else _json(record.error),
                _iso_timestamp(record.queued_at),
                None
                if record.started_at is None
                else _iso_timestamp(record.started_at),
                None
                if record.finished_at is None
                else _iso_timestamp(record.finished_at),
                _iso_timestamp(record.updated_at),
                record.job_id,
            ),
        )

    async def transition(
        self,
        job_id: str,
        *,
        expected: JobStatus,
        target: JobStatus,
        now: float,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        if not isinstance(expected, JobStatus) or not isinstance(target, JobStatus):
            raise ValueError("expected and target must be JobStatus values")
        timestamp = _finite_non_negative("now", now)
        async with self.database.transaction("IMMEDIATE") as connection:
            current = await self._load_for_update(connection, job_id)
            updated = _transition_record(
                current,
                expected=expected,
                target=target,
                now=timestamp,
                result=result,
                error=error,
            )
            await self._persist(connection, updated)
        return updated

    async def retry(
        self,
        job_id: str,
        *,
        now: float,
        expected: JobStatus = JobStatus.FAILED,
        error: Mapping[str, Any] | None = None,
    ) -> JobRecord:
        if not isinstance(expected, JobStatus):
            raise ValueError("expected must be JobStatus")
        timestamp = _finite_non_negative("now", now)
        async with self.database.transaction("IMMEDIATE") as connection:
            current = await self._load_for_update(connection, job_id)
            updated = _retry_record(
                current,
                expected=expected,
                now=timestamp,
                error=error,
            )
            await self._persist(connection, updated)
        return updated


__all__ = [
    "DuplicateJob",
    "InMemoryJobRepository",
    "InvalidJobTransition",
    "JobError",
    "JobNotFound",
    "JobRecord",
    "JobRepository",
    "JobStatus",
    "JobSubmission",
    "JobTransitionConflict",
    "SqliteJobRepository",
]
