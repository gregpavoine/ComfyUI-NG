CREATE TABLE models (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE,
    architecture TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'available'
        CHECK (status IN ('discovering', 'available', 'loading', 'loaded', 'offloaded', 'evicted', 'unsupported', 'failed')),
    capabilities_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(capabilities_json)),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) STRICT;

CREATE TABLE model_files (
    id TEXT PRIMARY KEY NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    format TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (model_id, path)
) STRICT;

CREATE TABLE model_sources (
    id TEXT PRIMARY KEY NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (model_id, provider, source_id, revision)
) STRICT;

CREATE TABLE loras (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE,
    model_family TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    status TEXT NOT NULL DEFAULT 'available'
        CHECK (status IN ('available', 'incompatible', 'missing', 'failed')),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) STRICT;

CREATE TABLE plugins (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'disabled'
        CHECK (status IN ('installing', 'enabled', 'disabled', 'failed', 'uninstalled')),
    isolation INTEGER NOT NULL DEFAULT 1 CHECK (isolation IN (0, 1)),
    current_version TEXT,
    permissions_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(permissions_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) STRICT;

CREATE TABLE plugin_versions (
    id TEXT PRIMARY KEY NOT NULL,
    plugin_id TEXT NOT NULL REFERENCES plugins(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'staged'
        CHECK (status IN ('staged', 'active', 'inactive', 'failed')),
    manifest_json TEXT NOT NULL CHECK (json_valid(manifest_json)),
    install_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (plugin_id, version)
) STRICT;

CREATE TABLE node_types (
    id TEXT PRIMARY KEY NOT NULL,
    type_id TEXT NOT NULL,
    version TEXT NOT NULL,
    plugin_version_id TEXT REFERENCES plugin_versions(id) ON DELETE CASCADE,
    schema_json TEXT NOT NULL CHECK (json_valid(schema_json)),
    resources_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(resources_json)),
    lifecycle TEXT NOT NULL DEFAULT 'per_job'
        CHECK (lifecycle IN ('per_call', 'per_job', 'warm', 'persistent')),
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (type_id, version)
) STRICT;

CREATE TABLE workflows (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    current_version INTEGER NOT NULL DEFAULT 0 CHECK (current_version >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) STRICT;

CREATE TABLE workflow_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    graph_json TEXT NOT NULL CHECK (json_valid(graph_json)),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (workflow_id, version),
    UNIQUE (id, workflow_id)
) STRICT;

CREATE TABLE jobs (
    id TEXT PRIMARY KEY NOT NULL,
    workflow_id TEXT NOT NULL,
    workflow_version_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'preparing', 'running', 'completed', 'failed', 'cancelled')),
    inputs_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(inputs_json)),
    execution_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(execution_json)),
    priority INTEGER NOT NULL DEFAULT 50 CHECK (priority BETWEEN 0 AND 100),
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 1 CHECK (max_attempts > 0),
    idempotency_key TEXT,
    request_hash TEXT,
    error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    queued_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (workflow_version_id, workflow_id)
        REFERENCES workflow_versions(id, workflow_id) ON DELETE RESTRICT,
    CHECK ((idempotency_key IS NULL) = (request_hash IS NULL))
) STRICT;

CREATE TABLE job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (job_id, sequence)
) STRICT;

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY NOT NULL,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    workflow_id TEXT REFERENCES workflows(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT CHECK (sha256 IS NULL OR length(sha256) = 64),
    size_bytes INTEGER NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (owner_type, owner_id, name, version)
) STRICT;

CREATE TABLE workers (
    id TEXT PRIMARY KEY NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'starting'
        CHECK (status IN ('starting', 'idle', 'busy', 'stopping', 'stopped', 'crashed', 'unhealthy')),
    pid INTEGER CHECK (pid IS NULL OR pid > 0),
    capabilities_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(capabilities_json)),
    last_heartbeat_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) STRICT;

CREATE TABLE benchmarks (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    config_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(config_json)),
    results_json TEXT CHECK (results_json IS NULL OR json_valid(results_json)),
    environment_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(environment_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at TEXT,
    completed_at TEXT
) STRICT;

CREATE TABLE provider_accounts (
    id TEXT PRIMARY KEY NOT NULL,
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'enabled'
        CHECK (status IN ('enabled', 'disabled', 'unhealthy')),
    config_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(config_json)),
    secret_ref TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (provider, name)
) STRICT;

CREATE TABLE downloads (
    id TEXT PRIMARY KEY NOT NULL,
    provider_account_id TEXT REFERENCES provider_accounts(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'downloading', 'paused', 'completed', 'failed', 'cancelled')),
    url TEXT NOT NULL,
    target_path TEXT NOT NULL,
    bytes_complete INTEGER NOT NULL DEFAULT 0 CHECK (bytes_complete >= 0),
    bytes_total INTEGER CHECK (bytes_total IS NULL OR bytes_total >= 0),
    etag TEXT,
    checksum TEXT,
    error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT,
    CHECK (bytes_total IS NULL OR bytes_complete <= bytes_total)
) STRICT;

CREATE TABLE cache_entries (
    key TEXT PRIMARY KEY NOT NULL,
    namespace TEXT NOT NULL,
    value_ref TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    expires_at TEXT,
    last_accessed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    hit_count INTEGER NOT NULL DEFAULT 0 CHECK (hit_count >= 0)
) STRICT;

CREATE TABLE settings (
    key TEXT PRIMARY KEY NOT NULL,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
) STRICT;

CREATE INDEX idx_model_files_model_id ON model_files(model_id);
CREATE INDEX idx_model_files_sha256 ON model_files(sha256);
CREATE INDEX idx_model_sources_model_id ON model_sources(model_id);
CREATE INDEX idx_loras_family_status ON loras(model_family, status);
CREATE INDEX idx_plugin_versions_plugin_id ON plugin_versions(plugin_id);
CREATE INDEX idx_node_types_plugin_version_id ON node_types(plugin_version_id);
CREATE INDEX idx_node_types_type_id_enabled ON node_types(type_id, enabled);
CREATE INDEX idx_workflow_versions_workflow_id ON workflow_versions(workflow_id, version DESC);
CREATE INDEX idx_jobs_status_priority ON jobs(status, priority DESC, created_at);
CREATE INDEX idx_jobs_workflow_version_id ON jobs(workflow_version_id);
CREATE UNIQUE INDEX idx_jobs_idempotency_key ON jobs(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX idx_job_events_job_sequence ON job_events(job_id, sequence);
CREATE INDEX idx_artifacts_owner ON artifacts(owner_type, owner_id, name, version DESC);
CREATE INDEX idx_artifacts_job_id ON artifacts(job_id);
CREATE INDEX idx_artifacts_workflow_id ON artifacts(workflow_id);
CREATE INDEX idx_workers_status_kind ON workers(status, kind);
CREATE INDEX idx_benchmarks_status_created ON benchmarks(status, created_at);
CREATE INDEX idx_benchmarks_job_id ON benchmarks(job_id);
CREATE INDEX idx_provider_accounts_status ON provider_accounts(provider, status);
CREATE INDEX idx_downloads_status_created ON downloads(status, created_at);
CREATE INDEX idx_downloads_provider_account_id ON downloads(provider_account_id);
CREATE INDEX idx_cache_entries_namespace_accessed ON cache_entries(namespace, last_accessed_at);
CREATE INDEX idx_cache_entries_expires_at ON cache_entries(expires_at);
