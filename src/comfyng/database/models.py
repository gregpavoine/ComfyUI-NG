from __future__ import annotations

from dataclasses import dataclass


APPLICATION_TABLES = (
    "models",
    "model_files",
    "model_sources",
    "loras",
    "plugins",
    "plugin_versions",
    "node_types",
    "workflows",
    "workflow_versions",
    "jobs",
    "job_events",
    "artifacts",
    "workers",
    "benchmarks",
    "provider_accounts",
    "downloads",
    "cache_entries",
    "settings",
)


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    primary_key: str
    columns: frozenset[str]


def _spec(name: str, primary_key: str, *columns: str) -> TableSpec:
    return TableSpec(name, primary_key, frozenset((primary_key, *columns)))


TABLE_SPECS = {
    "models": _spec(
        "models", "id", "name", "architecture", "status", "capabilities_json",
        "metadata_json", "created_at", "updated_at",
    ),
    "model_files": _spec(
        "model_files", "id", "model_id", "kind", "path", "sha256", "size_bytes",
        "format", "metadata_json", "created_at",
    ),
    "model_sources": _spec(
        "model_sources", "id", "model_id", "provider", "source_id", "revision",
        "metadata_json", "created_at",
    ),
    "loras": _spec(
        "loras", "id", "name", "model_family", "path", "sha256", "status",
        "metadata_json", "created_at", "updated_at",
    ),
    "plugins": _spec(
        "plugins", "id", "name", "status", "isolation", "current_version",
        "permissions_json", "created_at", "updated_at",
    ),
    "plugin_versions": _spec(
        "plugin_versions", "id", "plugin_id", "version", "status", "manifest_json",
        "install_path", "created_at",
    ),
    "node_types": _spec(
        "node_types", "id", "type_id", "version", "plugin_version_id", "schema_json",
        "resources_json", "lifecycle", "enabled", "created_at",
    ),
    "workflows": _spec(
        "workflows", "id", "name", "description", "current_version", "metadata_json",
        "created_at", "updated_at",
    ),
    "workflow_versions": _spec(
        "workflow_versions", "id", "workflow_id", "version", "graph_json",
        "metadata_json", "created_at",
    ),
    "jobs": _spec(
        "jobs", "id", "workflow_id", "workflow_version_id", "status", "inputs_json",
        "execution_json", "priority", "attempt", "max_attempts", "idempotency_key",
        "request_hash", "error_json", "created_at", "queued_at", "started_at",
        "finished_at", "updated_at",
    ),
    "job_events": _spec(
        "job_events", "id", "job_id", "sequence", "event_type", "payload_json",
        "created_at",
    ),
    "artifacts": _spec(
        "artifacts", "id", "owner_type", "owner_id", "name", "version", "job_id",
        "workflow_id", "kind", "uri", "sha256", "size_bytes", "metadata_json",
        "created_at",
    ),
    "workers": _spec(
        "workers", "id", "kind", "status", "pid", "capabilities_json",
        "last_heartbeat_at", "created_at", "updated_at",
    ),
    "benchmarks": _spec(
        "benchmarks", "id", "name", "status", "job_id", "config_json", "results_json",
        "environment_json", "created_at", "started_at", "completed_at",
    ),
    "provider_accounts": _spec(
        "provider_accounts", "id", "provider", "name", "status", "config_json",
        "secret_ref", "created_at", "updated_at",
    ),
    "downloads": _spec(
        "downloads", "id", "provider_account_id", "status", "url", "target_path",
        "bytes_complete", "bytes_total", "etag", "checksum", "error_json", "created_at",
        "updated_at", "completed_at",
    ),
    "cache_entries": _spec(
        "cache_entries", "key", "namespace", "value_ref", "size_bytes", "metadata_json",
        "expires_at", "last_accessed_at", "created_at", "hit_count",
    ),
    "settings": _spec("settings", "key", "value_json", "updated_at"),
}


if tuple(TABLE_SPECS) != APPLICATION_TABLES:
    raise RuntimeError("table specifications must match the application table order")


class DatabaseError(RuntimeError):
    """Base class for durable-state errors."""


class MigrationError(DatabaseError):
    """Raised when migration metadata or the on-disk schema is invalid."""


class IdempotencyConflict(DatabaseError):
    """Raised when an idempotency key is reused for another request."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"idempotency key {key!r} was reused with a different request")
