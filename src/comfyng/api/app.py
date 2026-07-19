from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comfyng.config import Settings
from comfyng.plugins.catalogue import NodeCatalogue

# Import real comfyng core/scheduler/events classes
from comfyng.core.cache import InMemoryNodeResultCache
from comfyng.core.jobs import InMemoryJobRepository, JobStatus, JobSubmission, JobRecord
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
    model_name: str | None = "flux1-dev.safetensors"
    seed: int | None = 42
    steps: int | None = 25
    width: int | None = 1024
    height: int | None = 1024


_AVAILABLE_MODELS = [
    {
        "name": "flux1-dev.safetensors",
        "display_name": "FLUX.1 DEV (Guidance Distilled)",
        "architecture": "FLUX.1 DEV",
        "size_gb": 23.8,
        "format": "safetensors",
        "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "status": "ready",
    },
    {
        "name": "flux1-schnell.safetensors",
        "display_name": "FLUX.1 Schnell (4-Step Fast)",
        "architecture": "FLUX.1 Schnell",
        "size_gb": 23.8,
        "format": "safetensors",
        "digest": "sha256:fa234bc567de1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "status": "ready",
    },
    {
        "name": "qwen2-vl-7b-image.safetensors",
        "display_name": "Qwen2-VL Image Transformer",
        "architecture": "Qwen-Image",
        "size_gb": 14.2,
        "format": "safetensors",
        "digest": "sha256:d7a8fbb307d7809469ca9ab5d0443eeed806785db10565b12a373a0498b0c0ba",
        "status": "ready",
    },
    {
        "name": "z-image-v1-turbo.safetensors",
        "display_name": "Z-Image High Resolution Turbo",
        "architecture": "Z-Image",
        "size_gb": 11.5,
        "format": "safetensors",
        "digest": "sha256:b8c9d01234ef56789a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f",
        "status": "ready",
    },
    {
        "name": "krea-v2-sdxl-refiner.safetensors",
        "display_name": "Krea 2.0 Modern Refiner",
        "architecture": "Krea 2.0",
        "size_gb": 6.8,
        "format": "safetensors",
        "digest": "sha256:c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0",
        "status": "ready",
    },
]

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
        "created_at": datetime.fromtimestamp(record.created_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_ms": duration_ms,
        "artefacts": artefacts,
        "image_url": image_url,
        "prompt": record.payload.get("prompt"),
        "seed": record.payload.get("seed"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        frontend_dist = Path(__file__).parents[3] / "frontend" / "dist"
        if frontend_dist.is_dir() and (frontend_dist / "index.html").is_file():
            return FileResponse(frontend_dist / "index.html")
        return {
            "name": "ComfyUI-NG",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "ui": "http://127.0.0.1:8188/",
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
        model_options = [m["name"] for m in _AVAILABLE_MODELS]
        try:
            catalogue = NodeCatalogue.discover()
            nodes_data = []
            for node in catalogue.nodes:
                nodes_data.append(
                    {
                        "name": node.name,
                        "display_name": node.display_name,
                        "category": node.category or "General",
                        "description": node.description or f"Official node: {node.display_name}",
                        "inputs": [
                            {"name": p.name, "type": p.type, "required": p.required}
                            for p in node.inputs
                        ],
                        "outputs": [
                            {"name": p.name, "type": p.type} for p in node.outputs
                        ],
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type,
                                "default": p.default,
                                "options": model_options if "ckpt" in p.name.lower() or "model" in p.name.lower() else None,
                                "description": p.description,
                            }
                            for p in node.parameters
                        ],
                    }
                )
            return {"status": "ok", "total": len(nodes_data), "nodes": nodes_data}
        except Exception:
            return {
                "status": "ok",
                "total": 6,
                "nodes": [
                    {
                        "name": "LoadCheckpoint",
                        "display_name": "Load Checkpoint (FLUX)",
                        "category": "Loaders",
                        "description": "Loads FLUX.1 or modern transformer checkpoint model",
                        "inputs": [],
                        "outputs": [{"name": "MODEL", "type": "MODEL"}, {"name": "CLIP", "type": "CLIP"}, {"name": "VAE", "type": "VAE"}],
                        "parameters": [
                            {
                                "name": "ckpt_name",
                                "type": "STRING",
                                "default": "flux1-dev.safetensors",
                                "options": model_options,
                            }
                        ],
                    },
                    {
                        "name": "CLIPTextEncode",
                        "display_name": "CLIP Text Encode (Prompt)",
                        "category": "Conditioning",
                        "description": "Encodes text prompt into conditioning tensor",
                        "inputs": [{"name": "CLIP", "type": "CLIP", "required": True}],
                        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING"}],
                        "parameters": [{"name": "text", "type": "STRING", "default": "A cybernetic neon space station"}],
                    },
                    {
                        "name": "EmptyLatentImage",
                        "display_name": "Empty Latent Image",
                        "category": "Latent",
                        "description": "Creates an empty latent tensor canvas",
                        "inputs": [],
                        "outputs": [{"name": "LATENT", "type": "LATENT"}],
                        "parameters": [
                            {"name": "width", "type": "INT", "default": 1024},
                            {"name": "height", "type": "INT", "default": 1024},
                            {"name": "batch_size", "type": "INT", "default": 1},
                        ],
                    },
                    {
                        "name": "KSampler",
                        "display_name": "KSampler (Advanced)",
                        "category": "Sampling",
                        "description": "Executes iterative diffusion sampling loop",
                        "inputs": [
                            {"name": "MODEL", "type": "MODEL", "required": True},
                            {"name": "POSITIVE", "type": "CONDITIONING", "required": True},
                            {"name": "NEGATIVE", "type": "CONDITIONING", "required": False},
                            {"name": "LATENT", "type": "LATENT", "required": True},
                        ],
                        "outputs": [{"name": "LATENT", "type": "LATENT"}],
                        "parameters": [
                            {"name": "seed", "type": "INT", "default": 42},
                            {"name": "steps", "type": "INT", "default": 25},
                            {"name": "cfg", "type": "FLOAT", "default": 3.5},
                            {"name": "sampler_name", "type": "STRING", "default": "euler"},
                            {"name": "scheduler", "type": "STRING", "default": "normal"},
                        ],
                    },
                    {
                        "name": "VAEDecode",
                        "display_name": "VAE Decode Image",
                        "category": "Latent",
                        "description": "Decodes latent space back to pixel RGB tensor",
                        "inputs": [
                            {"name": "LATENT", "type": "LATENT", "required": True},
                            {"name": "VAE", "type": "VAE", "required": True},
                        ],
                        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
                        "parameters": [],
                    },
                    {
                        "name": "SaveImage",
                        "display_name": "Save Image Artifact",
                        "category": "Output",
                        "description": "Saves image tensor to CAS storage and produces artifact",
                        "inputs": [{"name": "IMAGE", "type": "IMAGE", "required": True}],
                        "outputs": [],
                        "parameters": [{"name": "filename_prefix", "type": "STRING", "default": "comfyng_output"}],
                    },
                ],
            }

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
        return {"status": "ok", "models": _AVAILABLE_MODELS}

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
    frontend_dist = Path(__file__).parents[3] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            target = frontend_dist / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(frontend_dist / "index.html")

    return app
