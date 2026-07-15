# ComfyUI-NG V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer le moteur ComfyUI-NG complet décrit par `docs/SPECIFICATION.md`, depuis l’installation jusqu’à l’éditeur nodal et au runtime ML JIT.

**Architecture:** Le control-plane FastAPI/SQLite reste léger et sans dépendance ML. Les graphes, jobs et ressources sont orchestrés par des services typés ; toute exécution lourde se fait dans des workers séparés, et le frontend React consomme exclusivement l’API v1 et ses flux événementiels.

**Tech Stack:** Python 3.14, FastAPI, Pydantic 2, msgspec, aiosqlite, structlog, prometheus-client, Typer, PyTorch 2.11+ en extra, diffusers/transformers en extra, React 19, TypeScript, Vite, React Flow, Vitest, Playwright.

## Global Constraints

- Python `>=3.14`; cible de production Linux x86_64 NVIDIA CUDA.
- API/core sans import de Torch, CUDA, diffusers, transformers ou provider au démarrage.
- `forkserver` sous Linux ; `spawn` sur macOS/Windows et pour tout runtime qui l’exige.
- Bind par défaut `127.0.0.1:8188`; aucun réseau requis au boot ou pendant une génération locale.
- Aucun support SD1.x, SD2.x, SDXL, checkpoints SD monolithiques ni latents historiques 4 canaux.
- Catalogue des 40 nodes officiels visible sans importer leurs runtimes.
- SQLite V1 en WAL, foreign keys actives et busy timeout explicite.
- Toutes les opérations longues retournent immédiatement un job asynchrone.
- WebSocket, SSE, idempotence, annulation, OpenAPI et artefacts/workflows versionnés sont obligatoires.
- Les providers sont des extras optionnels et ne sont jamais consultés par le sampler.
- Les fichiers lourds passent par CAS, mémoire partagée ou handles ; jamais dans un payload JSON IPC.
- Aucun bouton frontend sans action, aucune route factice et aucun placeholder fonctionnel.

---

### Task 1: Fondation reproductible et configuration stricte

**Files:**
- Create: `pyproject.toml`, `config/default.yaml`, `.python-version`, `.gitignore`, `LICENSE`, `README.md`
- Create: `src/comfyng/__init__.py`, `src/comfyng/config/{__init__.py,models.py,loader.py}`, `src/comfyng/cli/main.py`
- Test: `tests/unit/test_config.py`, `tests/architecture/test_core_imports.py`

**Interfaces:**
- Produces: `Settings`, `load_settings(path: Path | None, env: Mapping[str, str] | None) -> Settings`, console command `comfyng`.

- [ ] **Step 1: Write failing tests** for exact defaults, env precedence, invalid ports/budgets, Python floor and absence of heavy/provider modules in `sys.modules` after importing `comfyng`.
- [ ] **Step 2: Run** `python3.14 -m pytest tests/unit/test_config.py tests/architecture/test_core_imports.py -q`; expect failures because package/configuration do not exist.
- [ ] **Step 3: Implement** nested frozen Pydantic settings with `Settings.load`, `${ENV}` expansion, safe paths under a configurable data root and a Typer root command.
- [ ] **Step 4: Run** the focused tests, `python3.14 -m build`, and `comfyng --help`; expect success and all specified CLI groups visible.
- [ ] **Step 5: Commit** `feat: establish ComfyUI-NG foundation`.

### Task 2: SQLite WAL, migrations and repositories

**Files:**
- Create: `migrations/0001_initial.sql`, `src/comfyng/database/{connection.py,migrations.py,models.py,repositories.py}`
- Test: `tests/unit/test_database.py`, `tests/integration/test_database_concurrency.py`

**Interfaces:**
- Consumes: `Settings.database`.
- Produces: `Database.open()`, `Database.transaction()`, repositories for all 18 required tables, monotonic workflow/artifact versions.

- [ ] **Step 1: Write failing tests** asserting `journal_mode=wal`, `foreign_keys=1`, `busy_timeout=5000`, the exact table set, FK enforcement, version increments and concurrent reader/writer behavior.
- [ ] **Step 2: Run** the two database test modules; expect missing module/migration failures.
- [ ] **Step 3: Implement** one idempotent migration with explicit PK/FK/indexes/check constraints and async repositories using transactions for state transitions and idempotency records.
- [ ] **Step 4: Run** focused tests twice against a fresh and already-migrated DB; expect identical schema and passing concurrency checks.
- [ ] **Step 5: Commit** `feat: add durable SQLite state`.

### Task 3: Domain types, capabilities and declarative node catalogue

**Files:**
- Create: `src/comfyng/core/{enums.py,errors.py,ids.py}`, `src/comfyng/models/capabilities.py`, `src/comfyng/graph/types.py`
- Create: `src/comfyng/plugins/{manifest.py,catalogue.py}`, `schemas/nodes/*.json`, `runtimes/*/ng-node.toml`
- Test: `tests/unit/test_capabilities.py`, `tests/unit/test_manifests.py`, `tests/unit/test_node_catalogue.py`

**Interfaces:**
- Produces: frozen `ModelCapabilities`, `ModelHandle`, `TensorHandle`, `NodeInstance`, `Edge`, `Graph`, `NodeDefinition`, `PluginManifest`, `NodeCatalogue.discover()`.

- [ ] **Step 1: Write failing tests** for round-trip serialization, semver/schema validation, path containment, lifecycle transitions, duplicate node rejection and exactly 40 official display names discoverable without runtime import.
- [ ] **Step 2: Run** the three modules; expect import and catalogue failures.
- [ ] **Step 3: Implement** strict dataclasses/msgspec structs, a versioned type registry, JSON schema loading and TOML discovery that never resolves the Python entrypoint.
- [ ] **Step 4: Run** tests plus an import-sentinel plugin whose module raises if imported; discovery must pass without triggering it.
- [ ] **Step 5: Commit** `feat: define typed node catalogue`.

### Task 4: Graph compiler, subgraphs and cache keys

**Files:**
- Create: `src/comfyng/graph/{validation.py,compiler.py,topology.py,subgraphs.py,cache.py}`
- Test: `tests/unit/graph/test_validation.py`, `tests/unit/graph/test_compiler.py`, `tests/property/test_graphs.py`

**Interfaces:**
- Produces: `GraphCompiler.compile(graph, context) -> ExecutionPlan`, `ExecutionGroup`, `ExecutionStep`, `GraphDiagnostic`, stable `node_cache_key`.

- [ ] **Step 1: Write failing tests** for duplicate IDs, missing ports, incompatible types/versions, cycles, unused outputs, constants, deterministic topological order, parallel branches, fan-out/in, bounded loops and conditions.
- [ ] **Step 2: Run** focused graph tests; expect missing compiler failures.
- [ ] **Step 3: Implement** Kahn topological sorting, explicit control nodes for loops/conditions, subgraph expansion, critical-path/resource annotations and content-derived cache keys.
- [ ] **Step 4: Run** unit/property tests over generated DAGs; expect deterministic plans and rejection of all invalid graphs.
- [ ] **Step 5: Commit** `feat: compile typed execution graphs`.

### Task 5: Hardware probe, thread budgets and resource admission

**Files:**
- Create: `src/comfyng/resources/{hardware.py,threads.py,budgets.py,broker.py,pressure.py}`
- Test: `tests/unit/resources/test_broker.py`, `tests/integration/test_hardware_probe.py`

**Interfaces:**
- Produces: `HardwareInventory`, `ThreadBudget`, `ResourceEstimate`, `ResourceReservation`, `ResourceBroker.admit()` and pressure events.

- [ ] **Step 1: Write failing tests** for reserved cores/RAM/VRAM, minimum worker bound, one heavy GPU job, thread env vars, unsupported probe fields, reservation release and explicit over-budget decisions.
- [ ] **Step 2: Run** resource tests; expect missing broker.
- [ ] **Step 3: Implement** psutil/platform probe plus optional `nvidia-smi` probe, deterministic fallback order `offload -> quantize -> sequence -> reduce_batch -> reject`, and bounded thread allocation.
- [ ] **Step 4: Run** tests on the current non-NVIDIA host and fixture-driven NVIDIA inventories; both must pass without importing Torch.
- [ ] **Step 5: Commit** `feat: add resource-aware admission`.

### Task 6: IPC handles, worker processes and supervision

**Files:**
- Create: `src/comfyng/workers/{protocol.py,process.py,supervisor.py,heartbeat.py,shared_memory.py,sandbox.py}`
- Create: `src/comfyng/runtime/entrypoint.py`
- Test: `tests/integration/workers/test_spawn.py`, `test_crash_recovery.py`, `test_shared_memory.py`, `test_unload.py`

**Interfaces:**
- Produces: `WorkerCommand`, `WorkerEvent`, `WorkerSpec`, `WorkerSupervisor.start/stop/execute`, `SharedObjectStore` with owner leases.

- [ ] **Step 1: Write failing tests** proving start method, heartbeats, timeout, process-tree termination, crash isolation, shared-memory round trip, owner cleanup and memory recovery after worker exit.
- [ ] **Step 2: Run** worker integration tests; expect missing supervisor.
- [ ] **Step 3: Implement** multiprocessing contexts chosen by platform/type, framed msgspec IPC, cancellation commands, heartbeat watchdog, restart budget/circuit breaker and lease cleanup.
- [ ] **Step 4: Run** tests including a deliberately crashing plugin and hung child; API test process must remain alive and all PIDs/resources must be reclaimed.
- [ ] **Step 5: Commit** `feat: isolate supervised workers`.

### Task 7: Scheduler, jobs, event journal and node-result cache

**Files:**
- Create: `src/comfyng/scheduler/{queues.py,priority.py,scheduler.py,cancellation.py,retry.py}`
- Create: `src/comfyng/events/{bus.py,models.py,journal.py}`, `src/comfyng/core/jobs.py`, `src/comfyng/core/cache.py`
- Test: `tests/unit/scheduler/*`, `tests/integration/test_job_lifecycle.py`, `tests/integration/test_cancellation.py`

**Interfaces:**
- Produces: six queues, `CancellationToken`, `Scheduler.submit/cancel/retry/run`, monotonic `JobStatus`, replayable `EventEnvelope`.

- [ ] **Step 1: Write failing tests** for the exact priority formula, age anti-starvation, queue limits, cache reuse, backpressure, bounded retry, all state transitions and cancellation at every sampler checkpoint.
- [ ] **Step 2: Run** scheduler/job tests; expect missing scheduler.
- [ ] **Step 3: Implement** async scheduler loop, atomic repository transitions, resource reservations, worker dispatch, cache lookups and durable event publication.
- [ ] **Step 4: Run** deterministic virtual-clock tests and 100 concurrent synthetic jobs; expect no starvation, duplicate terminal state or leaked reservation.
- [ ] **Step 5: Commit** `feat: orchestrate asynchronous jobs`.

### Task 8: Content-addressed storage, artefacts and model detection

**Files:**
- Create: `src/comfyng/storage/{cas.py,imports.py,artifacts.py,locks.py,gc.py}`
- Create: `src/comfyng/models/{detector.py,registry.py,inspection.py,legacy.py}`
- Test: `tests/unit/storage/*`, `tests/unit/models/*`, `tests/integration/test_atomic_model_import.py`

**Interfaces:**
- Produces: `CAS.put/open/link/import_external`, `ArtifactStore`, `ArchitectureDetector.detect`, `ModelRegistry.import_model/evict`.

- [ ] **Step 1: Write failing tests** for SHA-256 deduplication under concurrency, atomic rename, all six import modes, partial cleanup, path traversal, model multifile publication and architecture evidence precedence.
- [ ] **Step 2: Run** storage/model tests; expect missing stores.
- [ ] **Step 3: Implement** per-digest locks, fsync + atomic replace, immutable manifests, reference tracking and detector evidence scoring from safetensors metadata/config/tensor shapes/provider declaration.
- [ ] **Step 4: Run** tests with renamed files and SD1/SD2/SDXL fixtures; names must not affect detection and legacy models must return `unsupported_model_generation` before GPU work.
- [ ] **Step 5: Commit** `feat: add CAS and modern model registry`.

### Task 9: Providers and resumable download manager

**Files:**
- Create: `src/comfyng/providers/{base.py,local.py,http_manifest.py,huggingface.py,civitai_red.py,manager.py}`
- Create: `src/comfyng/storage/downloads.py`
- Test: `tests/contracts/test_providers.py`, `tests/integration/test_download_resume.py`, `tests/architecture/test_provider_isolation.py`

**Interfaces:**
- Produces: complete `ModelProvider` protocol including health/capabilities/search/inspect/list/resolve/auth/download, `DownloadManager.submit/cancel/resume`.

- [ ] **Step 1: Write failing contract tests** against all four adapters, an HTTP Range server, HF offline cache fixtures, Civitai handshake variants, hash mismatch and cancellation.
- [ ] **Step 2: Run** provider tests with only core extras; architecture test must first fail because adapters are absent, not because core imports provider packages.
- [ ] **Step 3: Implement** lazy adapter factories, bounded HTTP client, SSRF-safe URL validation, resumable part manifests, rate limits, progress events and CAS commit after hash verification.
- [ ] **Step 4: Run** tests with outbound network blocked except local fixture server; offline/local flows must pass and no provider module may appear during core boot.
- [ ] **Step 5: Commit** `feat: add optional model providers`.

### Task 10: Plugin installation, permissions and JIT lifecycle

**Files:**
- Create: `src/comfyng/plugins/{installer.py,environments.py,permissions.py,lifecycle.py,worker.py,signatures.py}`
- Test: `tests/unit/plugins/*`, `tests/integration/plugins/test_atomic_install.py`, `test_permission_enforcement.py`, `test_jit_unload.py`

**Interfaces:**
- Produces: `PluginInstaller.install`, `PermissionSet`, `PluginRuntimeManager.load/execute/unload`, all specified lifecycle/load policies.

- [ ] **Step 1: Write failing tests** injecting failure at each of the ten install phases, plus network/filesystem/subprocess denial, import sentinel, crash isolation and unload cleanup.
- [ ] **Step 2: Run** plugin tests; expect missing installer/runtime.
- [ ] **Step 3: Implement** staging environments, lock/hash/signature verification, atomic registration, permission handshake, worker-per-trust-group policy and JIT state machine.
- [ ] **Step 4: Run** malicious fixture plugins; core/API must survive and forbidden accesses must produce structured permission errors.
- [ ] **Step 5: Commit** `feat: add isolated JIT plugins`.

### Task 11: LoRA, sampling contracts and modern ML runtimes

**Files:**
- Create: `src/comfyng/lora/{models.py,validation.py,stack.py,cache.py}`
- Create: `src/comfyng/sampling/{config.py,compatibility.py,compile_cache.py,benchmark.py}`
- Create: `runtimes/{flux,qwen_image,z_image,krea2,core_image}/src/*`
- Test: `tests/unit/lora/*`, `tests/unit/sampling/*`, `tests/runtime/test_runtime_contract.py`, `tests/gpu/test_real_generation.py`

**Interfaces:**
- Produces: `LoraStack`, `PatchedRuntimeKey`, `SamplerConfig`, family-specific `RuntimeFactory.create`, `generate`, `apply_lora`, `keep_warm`, `evict`.

- [ ] **Step 1: Write failing tests** for LoRA family rejection before allocation, capability-filtered samplers/schedulers, compile-key components, seed determinism, T2I/I2I/inpaint routing, step cancellation and keep-warm/eviction.
- [ ] **Step 2: Run** unit/runtime contract tests with a deterministic in-test backend; expect missing runtime contracts.
- [ ] **Step 3: Implement** capability descriptors for all specified modern families and lazy diffusers adapters that import Torch only inside runtime worker entrypoints; support local files/directories and LoRA patching.
- [ ] **Step 4: Run** contract tests and the optional GPU test when `COMFYNG_TEST_MODEL` is defined; otherwise collect it as an explicit environment-gated test, not a silent pass.
- [ ] **Step 5: Commit** `feat: add modern isolated ML runtimes`.

### Task 12: Complete API v1, authentication, streams and webhooks

**Files:**
- Create: `src/comfyng/api/{app.py,dependencies.py,errors.py,idempotency.py,auth.py,streams.py}`
- Create: `src/comfyng/api/routes/{system,hardware,health,models,loras,nodes,plugins,workflows,jobs,artifacts,cache,providers,downloads,events,config,benchmarks,webhooks}.py`
- Test: `tests/api/*`, `tests/integration/test_event_streams.py`, `tests/integration/test_webhooks.py`

**Interfaces:**
- Produces: all `/api/v1` domains, `/api/v1/openapi.json`, `/docs`, `/redoc`, `/metrics`, WS/SSE replay and HMAC webhooks.

- [ ] **Step 1: Write failing API tests** for CRUD/list/detail/actions, pagination, canonical error envelope, API key/JWT, localhost default, idempotency replay/conflict, OpenAPI coverage, WS/SSE ordering/reconnect and webhook signatures/retries.
- [ ] **Step 2: Run** API tests; expect missing app/routes.
- [ ] **Step 3: Implement** lifespan wiring, typed Pydantic DTOs, auth dependencies, idempotency request hashes/24h TTL, cursor pagination, event envelopes and durable webhook deliveries with bounded exponential backoff.
- [ ] **Step 4: Run** API/integration tests and assert every declared path has a non-empty success and error schema in OpenAPI.
- [ ] **Step 5: Commit** `feat: expose complete API v1`.

### Task 13: CLI, logs, metrics, diagnostics and benchmarks

**Files:**
- Create: `src/comfyng/cli/{serve.py,doctor.py,models.py,plugins.py,jobs.py,cache.py,workers.py,benchmark.py}`
- Create: `src/comfyng/telemetry/{logging.py,metrics.py,tracing.py}`, `benchmarks/*.py`
- Test: `tests/cli/test_commands.py`, `tests/unit/test_metrics.py`, `tests/performance/test_startup.py`

**Interfaces:**
- Produces: every specified CLI command, JSON/human output, stable exit codes, structured logs, Prometheus metrics and optional OTel spans.

- [ ] **Step 1: Write failing CLI/telemetry tests** for command help and real actions, redaction, parseable JSON logs, bounded metric labels, all required metric families and boot/import/RAM probes.
- [ ] **Step 2: Run** the focused tests; expect command/module failures.
- [ ] **Step 3: Implement** thin CLI clients/services, doctor checks, benchmark persistence, structlog processors, Prometheus instruments and optional OpenTelemetry setup.
- [ ] **Step 4: Run** all CLI commands against a temporary live server and execute startup benchmarks; CUDA/network/heavy modules must remain absent at idle.
- [ ] **Step 5: Commit** `feat: add operations and observability`.

### Task 14: Frontend foundation and real-time data layer

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/src/{main.tsx,App.tsx,styles.css}`
- Create: `frontend/src/api/{client.ts,types.ts,events.ts,openapi.ts}`, `frontend/src/state/*`, `frontend/src/components/shell/*`
- Test: `frontend/src/**/*.test.tsx`

**Interfaces:**
- Consumes: API v1/OpenAPI/WS/SSE.
- Produces: responsive application shell, routing, query cache, reconnecting event client, accessible command palette and error boundary.

- [ ] **Step 1: Write failing Vitest tests** for navigation, offline/error states, WS→SSE fallback, event deduplication, reconnect cursor and keyboard navigation.
- [ ] **Step 2: Run** `pnpm test`; expect missing frontend.
- [ ] **Step 3: Implement** the visual system and app shell with API-derived types, persistent preferences and no mocked production data.
- [ ] **Step 4: Run** typecheck, unit tests and production build; expect zero TypeScript errors and a deployable `dist`.
- [ ] **Step 5: Commit** `feat: establish NG frontend`.

### Task 15: Nodal editor, workflow versioning and live validation

**Files:**
- Create: `frontend/src/features/editor/*`, `frontend/src/features/workflows/*`, `frontend/src/features/validation/*`
- Test: `frontend/src/features/editor/*.test.tsx`, `frontend/e2e/workflow.spec.ts`

**Interfaces:**
- Produces: node search/drag/connect/delete/duplicate, workflow tabs/save/version, inspector, typed connection validation and compile diagnostics.

- [ ] **Step 1: Write failing component/E2E tests** for catalogue loading without runtime, node search, valid/invalid edges, save/version/reopen, tabs, missing nodes, incompatible model/LoRA/VAE/scheduler/resources and job submission.
- [ ] **Step 2: Run** focused Vitest/Playwright tests; expect missing features.
- [ ] **Step 3: Implement** React Flow custom nodes and handles from schemas, palette, inspector controls, undo/redo, autosave guard and backend validation integration.
- [ ] **Step 4: Run** unit/E2E tests against a live temporary backend; expect a workflow created in UI to execute and expose an artefact.
- [ ] **Step 5: Commit** `feat: add typed workflow editor`.

### Task 16: Operational frontend surfaces

**Files:**
- Create: `frontend/src/features/{dashboard,jobs,models,loras,downloads,plugins,workers,monitoring,logs,benchmarks,api-explorer}/*`
- Test: `frontend/e2e/operations.spec.ts`, feature component tests.

**Interfaces:**
- Produces: all 14 required surfaces with real list/detail/action flows and live metrics/events.

- [ ] **Step 1: Write failing tests** for queue priority/cancel/retry, model import/keep-warm/evict, LoRA inspection, download pause/cancel/resume, plugin install/disable/unload, worker stop, log filters, benchmark launch and OpenAPI explorer requests.
- [ ] **Step 2: Run** frontend tests; expect missing pages/actions.
- [ ] **Step 3: Implement** every surface with shared tables/cards/confirmations/progress visualizations and keyboard/accessibility support.
- [ ] **Step 4: Run** full frontend unit/E2E/build suite and axe smoke tests; every visible action must change server state or provide a clear disabled reason.
- [ ] **Step 5: Commit** `feat: complete operator console`.

### Task 17: Packaging, first-run workflow and exhaustive verification

**Files:**
- Create: `packaging/{Dockerfile,docker-compose.yml,comfyng.service,install.sh}`, `scripts/{dev.py,verify.py}`
- Create: `tests/acceptance/test_v1_criteria.py`, `tests/soak/test_soak.py`, `docs/{INSTALL.md,USER_GUIDE.md,API.md,ARCHITECTURE.md,SECURITY.md,TEST_RESULTS.md}`
- Modify: `README.md`

**Interfaces:**
- Produces: one-command development start, NVIDIA production image, systemd service, acceptance matrix and reproducible verification report.

- [ ] **Step 1: Write failing acceptance tests** mapping all 20 V1 criteria plus WS/SSE/idempotence/offline boot/frontend/CAS/plugin invariants to executable probes.
- [ ] **Step 2: Run** the complete verification script; record all failing checks.
- [ ] **Step 3: Implement** packaging, demo workflow using the test runtime only in test mode, Linux GPU runner, 1,000-job/24h soak harness and complete operator documentation.
- [ ] **Step 4: Run** Python lint/type/tests, frontend lint/type/tests/build/E2E, package install smoke, server/CLI smoke, import/network guards, reduced soak and performance probes; fix every reproducible failure.
- [ ] **Step 5: Commit** `chore: package and verify ComfyUI-NG V1`.

### Task 18: Architecture graph and final audit

**Files:**
- Create: `graphify-out/*` through Graphify; update `docs/TEST_RESULTS.md` only with fresh evidence.

**Interfaces:**
- Produces: queryable architecture graph and final proof that module dependencies respect process/runtime/provider boundaries.

- [ ] **Step 1: Run Graphify** over the completed repository and inspect god nodes, surprising connections and dependency paths.
- [ ] **Step 2: Query** for any path from API/core boot modules to Torch, diffusers, transformers, huggingface_hub or Civitai code; expect no eager-import path.
- [ ] **Step 3: Run** `scripts/verify.py --all` from a clean environment and archive exact versions/results.
- [ ] **Step 4: Review** every requirement line in `docs/SPECIFICATION.md` against code, test or documented hardware-gated procedure; close every gap.
- [ ] **Step 5: Commit** `docs: record final architecture and validation`.

## Plan self-review

- All 38 specification sections map to at least one task.
- The 40-node catalogue is implemented and tested in Task 3, with runtime behavior covered by Tasks 4, 7, 8, 10 and 11.
- All 18 database tables, all public API domains, the six scheduler queues, the required CLI commands and all frontend functions are explicitly covered.
- CUDA/GPU and 24-hour checks are not claimed on the current macOS host; their executable Linux runners are a required deliverable.
- Provider optionality and the Hugging Face acceptance criterion are reconciled: the adapter is fully implemented as an installable extra, while local runtime operation never requires it.
- “GPU centralisé” means GPU ownership is restricted to controlled GPU workers; auxiliary GPU workers remain possible under resource admission.

