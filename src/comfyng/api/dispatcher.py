from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from comfyng.core.jobs import JobRecord
from comfyng.scheduler.cancellation import CancellationToken
from comfyng.scheduler.scheduler import DispatchResult
from comfyng.runtime.generator import generate_workflow_image


class WorkflowDispatcher:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    async def dispatch(
        self,
        job: JobRecord,
        token: CancellationToken,
    ) -> DispatchResult:
        payload = job.payload
        prompt = payload.get("prompt") or "A cybernetic space station surrounded by glowing neon plasma rings"
        seed = payload.get("seed", 42)
        steps = payload.get("steps", 25)
        width = payload.get("width", 1024)
        height = payload.get("height", 1024)
        model_name = payload.get("model_name") or "flux1-dev.safetensors"

        # Simulate some generation work (1 second sleep)
        await asyncio.sleep(1.0)
        token.checkpoint("sampling")

        # Run actual image generation and save to directory
        artifact = generate_workflow_image(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=3.5,
            model_name=model_name,
            storage_dir=self.artifacts_dir,
        )

        image_url = f"/api/v1/artifacts/{artifact['filename']}"

        return DispatchResult(
            value={
                "image_url": image_url,
                "filename": artifact["filename"],
                "digest": artifact["digest"],
                "width": width,
                "height": height,
                "prompt": prompt,
                "seed": seed,
            },
            cacheable=True,
            size_bytes=artifact["size_bytes"],
        )

    async def cancel(self, job_id: str) -> None:
        pass
