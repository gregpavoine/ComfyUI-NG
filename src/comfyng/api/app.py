from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any
import logging
import uuid

logger = logging.getLogger("comfyng.api.app")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comfyng.config import Settings
from comfyng.plugins.catalogue import NodeCatalogue

# Import real comfyng core/scheduler/events classes
from comfyng.core.cache import InMemoryNodeResultCache
from comfyng.core.jobs import InMemoryJobRepository, JobSubmission, JobRecord
from comfyng.events.bus import EventBus
from comfyng.events.journal import InMemoryEventJournal
from comfyng.resources.broker import ResourceBroker
from comfyng.resources.hardware import probe_hardware
from comfyng.scheduler.retry import RetryPolicy
from comfyng.scheduler.scheduler import Scheduler
from comfyng.api.dispatcher import WorkflowDispatcher


class JobSubmissionDTO(BaseModel):
    workflow_id: str = "workflow-1"
    name: str = "Generation Job"
    priority: int = 80
    prompt: str | None = None
    model_name: str | None = None
    seed: int | None = 42
    steps: int | None = 25
    width: int | None = 1024
    height: int | None = 1024
    nodes: list[dict[str, Any]] | None = None
    connections: list[dict[str, Any]] | None = None


def _schema_type(prop: Mapping[str, Any]) -> str:
    raw = prop.get("type")
    if isinstance(raw, str):
        return raw.upper()
    if isinstance(raw, (list, tuple)):
        for t in raw:
            if isinstance(t, str) and t != "null":
                return t.upper()
    return "ANY"


_AVAILABLE_MODELS: list[dict[str, Any]] = []

def _scan_files_in_dirs(base_dirs: list[Path], extensions: tuple[str, ...]) -> list[str]:
    files = []
    seen = set()
    for sdir in base_dirs:
        if not sdir.is_dir():
            continue
        try:
            for path in sdir.rglob("*"):
                if path.is_file() and path.suffix in extensions:
                    try:
                        rel_path = path.relative_to(sdir)
                    except ValueError:
                        rel_path = path.name
                    path_str = str(rel_path)
                    if path_str not in seen:
                        seen.add(path_str)
                        files.append(path_str)
        except Exception:
            pass
    files.sort()
    return files


def _get_real_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    for value in os.environ.get("COMFYNG_MODEL_PATHS", "").split(os.pathsep):
        v = value.strip()
        if v:
            dirs.append(Path(v).expanduser())
    home = Path.home()
    dirs.extend([
        home / ".local" / "share" / "comfyui-ng" / "models",
        home / "ComfyUI" / "models" / "diffusion_models",
        home / "ComfyUI" / "models" / "checkpoints",
        home / "ComfyUI" / "Models" / "diffusion_models",
        home / "ComfyUI" / "Models" / "checkpoints",
        home / "Documents" / "ComfyUI" / "models" / "diffusion_models",
        home / "Documents" / "ComfyUI" / "models" / "checkpoints",
    ])
    seen = set()
    result: list[Path] = []
    for d in dirs:
        resolved = d.resolve()
        if resolved.is_dir() and str(resolved) not in seen:
            seen.add(str(resolved))
            result.append(resolved)
    return result

def _get_real_models() -> list[dict[str, Any]]:
    import hashlib
    search_dirs = _get_real_model_dirs()
    models = []
    seen_paths = set()
    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        try:
            for path in sdir.rglob("*"):
                if path.is_file() and path.suffix in (".safetensors", ".ckpt"):
                    try:
                        rel_path = path.relative_to(sdir)
                    except ValueError:
                        rel_path = path.name
                    path_str = str(rel_path)
                    if path_str in seen_paths:
                        continue
                    seen_paths.add(path_str)
                    size_gb = round(path.stat().st_size / (1024**3), 1)
                    
                    parts = list(rel_path.parts)
                    if len(parts) >= 2:
                        category = parts[0]
                        name = parts[-1]
                        display_name = f"{name} ({category})"
                    else:
                        display_name = rel_path.name
                    
                    models.append({
                        "name": path_str,
                        "display_name": display_name,
                        "architecture": "FLUX" if "flux" in path_str.lower() else "SDXL" if "sdxl" in path_str.lower() else "Unknown",
                        "size_gb": size_gb,
                        "format": path.suffix[1:],
                        "digest": f"sha256:{hashlib.sha256(path_str.encode()).hexdigest()[:16]}",
                        "status": "ready",
                    })
        except Exception:
            pass
            
    models.sort(key=lambda m: m["display_name"])
    
    return models

_IN_MEMORY_WORKFLOWS: list[dict[str, Any]] = [
    {
        "id": "wf-flux-t2i",
        "name": "FLUX.1 Text-to-Image Standard",
        "description": "Standard FLUX.1 DEV sampling graph with text prompt and latent preview.",
        "nodes_count": 6,
        "updated_at": "2026-07-20T00:05:00Z",
    },
    {
        "id": "wf-qwen-i2i",
        "name": "Qwen-Image Image-to-Image Refiner",
        "description": "Conditioned image refiner with LoRA stack strength controls.",
        "nodes_count": 8,
        "updated_at": "2026-07-20T00:08:00Z",
    },
]


def _get_artifacts_dir_from_settings(settings: Settings | None) -> Path:
    if settings:
        artifacts_dir = settings.storage.root / "artifacts"
    else:
        artifacts_dir = Path.home() / ".local" / "share" / "comfyui-ng" / "storage" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def _format_job(record: JobRecord) -> dict[str, Any]:
    duration_ms = 0
    if record.finished_at is not None and record.started_at is not None:
        duration_ms = int((record.finished_at - record.started_at) * 1000)

    image_url = None
    artefacts = []
    if record.result and isinstance(record.result, dict):
        image_url = record.result.get("image_url")
        filename = record.result.get("filename")
        if filename:
            artefacts = [filename]

    return {
        "id": record.job_id,
        "name": record.payload.get("name") or "Generation Job",
        "status": record.status.value,
        "priority": record.user_priority,
        "created_at": datetime.fromtimestamp(time.time() - time.monotonic() + record.created_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_ms": duration_ms,
        "artefacts": artefacts,
        "image_url": image_url,
        "prompt": record.payload.get("prompt"),
        "seed": record.payload.get("seed"),
        "error": record.error["message"] if record.error else None,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan starting...")
    # Retrieve configuration settings
    settings: Settings | None = getattr(app.state, "settings", None)
    artifacts_dir = _get_artifacts_dir_from_settings(settings)

    # Initialize real services
    inventory = probe_hardware()
    broker = ResourceBroker(inventory=inventory)
    journal = InMemoryEventJournal()
    events = EventBus(journal)
    repository = InMemoryJobRepository()
    cache = InMemoryNodeResultCache()
    dispatcher = WorkflowDispatcher(artifacts_dir=artifacts_dir)

    scheduler = Scheduler(
        repository=repository,
        events=events,
        cache=cache,
        broker=broker,
        dispatcher=dispatcher,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=1),
    )

    # Save to app state
    app.state.repository = repository
    app.state.scheduler = scheduler
    app.state.events = events
    app.state.artifacts_dir = artifacts_dir

    # Start the background scheduler task
    scheduler_task = asyncio.create_task(scheduler.run())
    app.state.scheduler_task = scheduler_task

    logger.info("Lifespan yield reached!")
    yield

    # Stop scheduler and wait for completion
    scheduler.stop()
    try:
        await asyncio.wait_for(scheduler_task, timeout=3.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI control-plane application."""

    app = FastAPI(
        title="ComfyUI-NG API",
        description="A typed, local-first control plane for modern image-generation workflows.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )

    if settings is not None:
        app.state.settings = settings

    @app.get("/", response_model=None)
    async def root() -> dict[str, Any] | FileResponse:
        frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
        if frontend_dist.is_dir() and (frontend_dist / "index.html").is_file():
            return FileResponse(frontend_dist / "index.html")
        return {
            "name": "ComfyUI-NG",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "ui": "/",
        }

    @app.get("/health")
    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/system/info")
    async def system_info() -> dict[str, Any]:
        current_settings: Settings | None = getattr(app.state, "settings", None)
        return {
            "status": "ok",
            "version": "0.1.0",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "data_root": str(current_settings.data_root) if current_settings else "~/.local/share/comfyui-ng",
            "multiprocessing": "forkserver" if sys.platform != "win32" else "spawn",
            "active_workers": 2,
            "broker_status": "healthy",
        }

    @app.get("/api/v1/nodes/catalogue")
    async def get_node_catalogue() -> dict[str, Any]:
        model_dirs = _get_real_model_dirs()
        checkpoints = _scan_files_in_dirs(model_dirs, (".safetensors", ".ckpt"))
        
        vae_dirs: list[Path] = [p / "vae" for p in model_dirs] + [Path.home() / "ComfyUI" / "models" / "vae", Path.home() / "ComfyUI" / "Models" / "vae"]
        vaes = _scan_files_in_dirs([d for d in vae_dirs if d.exists()], (".safetensors", ".ckpt", ".pt"))
        
        lora_dirs: list[Path] = [p / "loras" for p in model_dirs] + [Path.home() / "ComfyUI" / "models" / "loras", Path.home() / "ComfyUI" / "Models" / "loras"]
        loras = _scan_files_in_dirs([d for d in lora_dirs if d.exists()], (".safetensors", ".ckpt"))

        catalogue = NodeCatalogue.discover()
        nodes_data = []
        for node in catalogue.nodes:
            inputs_schema = node.input_schema or {}
            outputs_schema = node.output_schema or {}
            in_props = inputs_schema.get("properties", {})
            out_props = outputs_schema.get("properties", {})
            required_inputs = set(inputs_schema.get("required", ()))

            nodes_data.append(
                {
                    "name": node.id,
                    "display_name": node.display_name,
                    "category": node.category or "General",
                    "description": node.description or f"Official node: {node.display_name}",
                    "inputs": [
                        {
                            "name": name,
                            "type": prop.get("x-comfyng-type", prop.get("type", "any")),
                            "required": name in required_inputs,
                        }
                        for name, prop in in_props.items()
                        if "x-comfyng-type" in prop
                    ],
                    "outputs": [
                        {
                            "name": name,
                            "type": prop.get("x-comfyng-type", prop.get("type", "any")),
                        }
                        for name, prop in out_props.items()
                    ],
                    "parameters": [
                        {
                            "name": name,
                            "type": _schema_type(prop),
                            "default": prop.get("default"),
                            "options": (
                                checkpoints if "ckpt" in name.lower() or "model" in name.lower()
                                else loras if "lora" in node.id.lower() or "lora" in name.lower()
                                else vaes if "vae" in node.id.lower() or "vae" in name.lower()
                                else None
                            ),
                            "description": prop.get("description"),
                        }
                        for name, prop in in_props.items()
                        if "x-comfyng-type" not in prop
                    ],
                }
            )
        return {"status": "ok", "total": len(nodes_data), "nodes": nodes_data}

    @app.get("/api/v1/jobs")
    async def list_jobs() -> dict[str, Any]:
        repository = getattr(app.state, "repository", None)
        if repository is None:
            return {"status": "ok", "jobs": []}
        records = await repository.list()
        # Sort so that newly created jobs are at the top
        sorted_records = sorted(records, key=lambda r: r.created_at, reverse=True)
        return {"status": "ok", "jobs": [_format_job(r) for r in sorted_records]}

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        repository = getattr(app.state, "repository", None)
        if repository is None:
            raise HTTPException(status_code=404, detail="Repository not initialized")
        record = await repository.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": "ok", "job": _format_job(record)}

    @app.post("/api/v1/jobs/submit")
    async def submit_job(job_req: JobSubmissionDTO) -> dict[str, Any]:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            raise HTTPException(status_code=503, detail="Scheduler not running")

        job_id = f"job-{int(time.time() * 1000) % 100000}-{uuid.uuid4().hex[:6]}"
        
        # Build core submission
        submission = JobSubmission(
            job_id=job_id,
            queue="normal",
            user_priority=job_req.priority,
            payload={
                "name": job_req.name,
                "prompt": job_req.prompt,
                "seed": job_req.seed,
                "steps": job_req.steps,
                "width": job_req.width,
                "height": job_req.height,
                "model_name": job_req.model_name,
                "nodes": job_req.nodes,
                "connections": job_req.connections,
            },
            workflow_id=job_req.workflow_id or "workflow-1",
            workflow_version_id=1,
        )

        record = await scheduler.submit(submission)
        return {"status": "ok", "message": "Job queued successfully", "job": _format_job(record)}

    @app.get("/api/v1/artifacts/{filename}", response_model=None)
    async def serve_artifact(filename: str) -> FileResponse:
        artifacts_dir = getattr(app.state, "artifacts_dir", None)
        if artifacts_dir is None:
            # Fallback path discovery
            settings: Settings | None = getattr(app.state, "settings", None)
            artifacts_dir = _get_artifacts_dir_from_settings(settings)
        
        file_path = artifacts_dir / filename
        if file_path.is_file():
            return FileResponse(file_path, media_type="image/png")

        raise HTTPException(status_code=404, detail="Artifact file not found")

    @app.get("/api/v1/workflows")
    async def list_workflows() -> dict[str, Any]:
        return {"status": "ok", "workflows": _IN_MEMORY_WORKFLOWS}

    @app.get("/api/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"status": "ok", "models": _get_real_models()}

    @app.get("/api/v1/plugins")
    async def list_plugins() -> dict[str, Any]:
        return {
            "status": "ok",
            "plugins": [
                {
                    "id": "official-nodes",
                    "name": "ComfyUI-NG Core Node Suite",
                    "version": "1.0.0",
                    "status": "enabled",
                    "permissions": {
                        "filesystem": "read-only",
                        "network": False,
                        "subprocess": False,
                    },
                },
                {
                    "id": "custom-sampler-ext",
                    "name": "Advanced Euler Schedulers",
                    "version": "0.2.1",
                    "status": "enabled",
                    "permissions": {
                        "filesystem": "sandboxed",
                        "network": False,
                        "subprocess": False,
                    },
                },
            ],
        }

    # Static assets serving for frontend
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            target = frontend_dist / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(frontend_dist / "index.html")

    return app
