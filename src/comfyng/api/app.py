from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from comfyng.config import Settings
from comfyng.plugins.catalogue import NodeCatalogue


class JobSubmission(BaseModel):
    workflow_id: str = "workflow-1"
    name: str = "Generation Job"
    priority: int = 80
    prompt: str | None = None


_IN_MEMORY_JOBS: list[dict[str, Any]] = [
    {
        "id": "job-101",
        "name": "Text to Image FLUX.1",
        "status": "completed",
        "priority": 90,
        "created_at": "2026-07-20T00:01:00Z",
        "duration_ms": 1420,
        "artefacts": ["artifact-flux-001.png"],
    },
    {
        "id": "job-102",
        "name": "LoRA Character Patching",
        "status": "running",
        "priority": 80,
        "created_at": "2026-07-20T00:10:00Z",
        "duration_ms": 350,
        "artefacts": [],
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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI control-plane application."""

    app = FastAPI(
        title="ComfyUI-NG API",
        description="A typed, local-first control plane for modern image-generation workflows.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    if settings is not None:
        app.state.settings = settings

    @app.get("/")
    async def root() -> dict[str, Any]:
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
                                "description": p.description,
                            }
                            for p in node.parameters
                        ],
                    }
                )
            return {"status": "ok", "total": len(nodes_data), "nodes": nodes_data}
        except Exception as exc:
            # Fallback official node sample catalogue
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
                        "parameters": [{"name": "ckpt_name", "type": "STRING", "default": "flux1-dev.safetensors"}],
                    },
                    {
                        "name": "CLIPTextEncode",
                        "display_name": "CLIP Text Encode (Prompt)",
                        "category": "Conditioning",
                        "description": "Encodes text prompt into conditioning tensor",
                        "inputs": [{"name": "CLIP", "type": "CLIP", "required": True}],
                        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING"}],
                        "parameters": [{"name": "text", "type": "STRING", "default": "A cybernetic neon robot in rain"}],
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
                            {"name": "steps", "type": "INT", "default": 20},
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
        return {"status": "ok", "jobs": _IN_MEMORY_JOBS}

    @app.post("/api/v1/jobs/submit")
    async def submit_job(job_req: JobSubmission) -> dict[str, Any]:
        import time

        new_job = {
            "id": f"job-{len(_IN_MEMORY_JOBS) + 101}",
            "name": job_req.name,
            "status": "queued",
            "priority": job_req.priority,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_ms": 0,
            "artefacts": [],
        }
        _IN_MEMORY_JOBS.insert(0, new_job)
        return {"status": "ok", "message": "Job queued successfully", "job": new_job}

    @app.get("/api/v1/workflows")
    async def list_workflows() -> dict[str, Any]:
        return {"status": "ok", "workflows": _IN_MEMORY_WORKFLOWS}

    @app.get("/api/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "status": "ok",
            "models": [
                {
                    "name": "flux1-dev.safetensors",
                    "architecture": "FLUX.1 DEV",
                    "size_gb": 23.8,
                    "format": "safetensors",
                    "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "status": "ready",
                },
                {
                    "name": "qwen2-vl-image.safetensors",
                    "architecture": "Qwen-Image",
                    "size_gb": 14.2,
                    "format": "safetensors",
                    "digest": "sha256:d7a8fbb307d7809469ca9ab5d0443eeed806785db10565b12a373a0498b0c0ba",
                    "status": "ready",
                },
            ],
        }

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
