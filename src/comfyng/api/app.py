from __future__ import annotations

import sys
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from comfyng.config import Settings


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
    async def root() -> dict[str, str]:
        return {
            "name": "ComfyUI-NG",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
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
            "data_root": str(current_settings.data_root) if current_settings else None,
        }

    return app
