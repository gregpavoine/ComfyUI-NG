from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

from comfyng.core.jobs import JobRecord
from comfyng.scheduler.cancellation import CancellationCheckpoint, CancellationToken
from comfyng.scheduler.scheduler import DispatchResult

logger = logging.getLogger("comfyng.api.dispatcher")


class NativeRuntimeError(RuntimeError):
    pass


class WorkflowDispatcher:
    """Native ComfyUI-NG dispatcher.

    This class intentionally contains no HTTP bridge to legacy ComfyUI. Jobs are
    executed directly by the ComfyUI-NG FLUX runtime.
    """

    def __init__(self, artifacts_dir: Path | str, runtime: object | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._runtime = runtime
        self._runtime_lock = asyncio.Lock()
        self._active_cancellations: dict[str, threading.Event] = {}

    def _load_runtime(self) -> object:
        if self._runtime is not None:
            return self._runtime
        project_root = Path(__file__).resolve().parents[3]
        plugin_src = project_root / "plugins" / "flux" / "src"
        if not plugin_src.is_dir():
            raise NativeRuntimeError(f"FLUX runtime package not found: {plugin_src}")
        sys.path.insert(0, str(plugin_src))
        try:
            module = importlib.import_module("comfyng_flux.runtime")
            self._runtime = module.FluxRuntime()
        finally:
            try:
                sys.path.remove(str(plugin_src))
            except ValueError:
                pass
        return self._runtime

    @staticmethod
    def _node_name(node: Mapping[str, Any]) -> str:
        definition = node.get("def")
        if isinstance(definition, Mapping):
            value = definition.get("name")
            if isinstance(value, str):
                return value
        for key in ("type", "class_type", "name"):
            value = node.get(key)
            if isinstance(value, str):
                return value
        return ""

    @staticmethod
    def _params(node: Mapping[str, Any]) -> Mapping[str, Any]:
        for key in ("params", "inputs", "properties"):
            value = node.get(key)
            if isinstance(value, Mapping):
                return value
        return {}

    @classmethod
    def _extract_request(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": payload.get("model_path") or payload.get("model_name"),
            "prompt": payload.get("prompt") or "",
            "negative_prompt": payload.get("negative_prompt") or "",
            "seed": int(payload.get("seed") or 42),
            "steps": int(payload.get("steps") or 25),
            "width": int(payload.get("width") or 1024),
            "height": int(payload.get("height") or 1024),
            "guidance": float(payload.get("guidance") or 3.5),
            "num_images": int(payload.get("num_images") or 1),
            "loras": list(payload.get("loras") or []),
            "model_family": payload.get("model_family", "auto"),
            "component_source": payload.get("component_source") or payload.get("components_path"),
        }
        nodes = payload.get("nodes")
        if not isinstance(nodes, (list, tuple)):
            return request

        for raw_node in nodes:
            if not isinstance(raw_node, Mapping):
                continue
            name = cls._node_name(raw_node).lower()
            params = cls._params(raw_node)
            if name in {"loadcheckpoint", "checkpointloadersimple", "ng.model.load", "ng.model.load"}:
                request["model"] = (
                    params.get("model_path")
                    or params.get("ckpt_name")
                    or params.get("model")
                    or request["model"]
                )
            elif name in {"cliptextencode", "ng.prompt.encode", "promptencode"}:
                text = params.get("text") or params.get("prompt")
                if isinstance(text, str) and text:
                    request["prompt"] = text
            elif name in {"emptylatentimage", "ng.empty_latent", "ng.latent.empty"}:
                request["width"] = int(params.get("width") or request["width"])
                request["height"] = int(params.get("height") or request["height"])
            elif name in {"ksampler", "ng.sample.run", "ng.sampler"}:
                request["seed"] = int(params.get("seed") or request["seed"])
                request["steps"] = int(params.get("steps") or request["steps"])
                request["guidance"] = float(
                    params.get("guidance") or params.get("cfg") or request["guidance"]
                )
            elif name in {"lora_load", "loraloader", "ng.lora.load"}:
                path = params.get("path") or params.get("lora_name") or params.get("lora")
                if path:
                    request["loras"].append(
                        {
                            "path": path,
                            "adapter_name": params.get("adapter_name"),
                            "model_strength": float(
                                params.get("model_strength")
                                or params.get("strength_model")
                                or params.get("strength")
                                or 1.0
                            ),
                        }
                    )
        return request

    @staticmethod
    def _search_roots(kind: str) -> tuple[Path, ...]:
        roots: list[Path] = []
        env_name = "COMFYNG_MODEL_PATHS" if kind == "model" else "COMFYNG_LORA_PATHS"
        for value in os.environ.get(env_name, "").split(os.pathsep):
            if value:
                roots.append(Path(value).expanduser())
        home = Path.home()
        if kind == "model":
            roots.extend(
                [
                    home / "ComfyUI-NG" / "models",
                    home / "ComfyUI" / "models" / "diffusion_models",
                    home / "ComfyUI" / "models" / "checkpoints",
                ]
            )
        else:
            roots.extend(
                [
                    home / "ComfyUI-NG" / "models" / "loras",
                    home / "ComfyUI" / "models" / "loras",
                ]
            )
        return tuple(path for path in roots if path.exists())

    @classmethod
    def _resolve_path(cls, value: object, *, kind: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise NativeRuntimeError(f"No {kind} was selected")
        candidate = Path(value).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        for root in cls._search_roots(kind):
            direct = root / value
            if direct.exists():
                return str(direct.resolve())
            try:
                matches = tuple(root.rglob(Path(value).name))
            except (OSError, PermissionError):
                continue
            if len(matches) == 1:
                return str(matches[0].resolve())
            if len(matches) > 1:
                raise NativeRuntimeError(
                    f"Ambiguous {kind} name {value!r}; use an absolute path. Matches: "
                    + ", ".join(str(item) for item in matches[:8])
                )
        searched = ", ".join(str(path) for path in cls._search_roots(kind)) or "(none)"
        raise NativeRuntimeError(f"{kind.capitalize()} {value!r} not found. Searched: {searched}")

    async def _execute_runtime(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        runtime = self._load_runtime()
        execute = getattr(runtime, "execute", None)
        if not callable(execute):
            raise NativeRuntimeError("FLUX runtime has no execute() method")
        return await asyncio.to_thread(execute, operation, payload, cancellation)

    async def dispatch(self, job: JobRecord, token: CancellationToken) -> DispatchResult:
        request = self._extract_request(job.payload)
        model_source = self._resolve_path(request["model"], kind="model")
        cancellation = threading.Event()
        self._active_cancellations[job.job_id] = cancellation

        async def watch_cancel() -> None:
            await token.wait()
            cancellation.set()

        watcher = asyncio.create_task(watch_cancel())
        try:
            async with self._runtime_lock:
                token.checkpoint(CancellationCheckpoint.BETWEEN_BLOCKS)
                load_result = await self._execute_runtime(
                    "ng.model.load",
                    {
                        "model_path": model_source,
                        "local_files_only": True,
                        "device": "auto",
                        "cpu_offload": True,
                        "compile_mode": "off",
                        "model_family": request.get("model_family", "auto"),
                        "component_source": request.get("component_source"),
                    },
                    cancellation,
                )

                loaded_adapters: list[dict[str, Any]] = []
                for index, raw_lora in enumerate(request["loras"]):
                    if isinstance(raw_lora, str):
                        raw_lora = {"path": raw_lora}
                    if not isinstance(raw_lora, Mapping):
                        raise NativeRuntimeError(f"Invalid LoRA entry at index {index}")
                    path = self._resolve_path(raw_lora.get("path"), kind="lora")
                    adapter_name = raw_lora.get("adapter_name") or f"lora_{index + 1}"
                    weight = float(raw_lora.get("model_strength") or raw_lora.get("weight") or 1.0)
                    result = await self._execute_runtime(
                        "ng.lora.load",
                        {"path": path, "adapter_name": adapter_name, "model_strength": weight},
                        cancellation,
                    )
                    loaded_adapters.append(dict(result))

                token.checkpoint(CancellationCheckpoint.SAMPLER_STEP, position=0)
                generated = await self._execute_runtime(
                    "ng.sample.run",
                    {
                        "prompt": request["prompt"],
                        "negative_prompt": request["negative_prompt"],
                        "seed": request["seed"],
                        "steps": request["steps"],
                        "width": request["width"],
                        "height": request["height"],
                        "guidance": request["guidance"],
                        "num_images": request["num_images"],
                    },
                    cancellation,
                )

                token.checkpoint(CancellationCheckpoint.BEFORE_SAVE)
                images = generated.get("images") if isinstance(generated, Mapping) else None
                if not isinstance(images, list) or not images:
                    raise NativeRuntimeError("Runtime returned no images")
                saved: list[dict[str, Any]] = []
                for index, image in enumerate(images):
                    if not isinstance(image, Mapping) or not isinstance(image.get("bytes"), bytes):
                        raise NativeRuntimeError("Runtime returned an invalid image payload")
                    filename = f"{job.job_id}-{index + 1}.png"
                    target = self.artifacts_dir / filename
                    await asyncio.to_thread(target.write_bytes, image["bytes"])
                    saved.append(
                        {
                            "filename": filename,
                            "image_url": f"/api/v1/artifacts/{filename}",
                            "width": image.get("width"),
                            "height": image.get("height"),
                            "digest": image.get("digest"),
                        }
                    )

                first = saved[0]
                value = {
                    "filename": first["filename"],
                    "image_url": first["image_url"],
                    "artifacts": saved,
                    "seed": generated.get("seed", request["seed"]),
                    "model": load_result,
                    "loras": loaded_adapters,
                    "engine": "comfyui-ng-native",
                }
                return DispatchResult(value=value, cacheable=False, size_bytes=sum((self.artifacts_dir / item["filename"]).stat().st_size for item in saved))
        finally:
            watcher.cancel()
            self._active_cancellations.pop(job.job_id, None)

    async def cancel(self, job_id: str) -> None:
        event = self._active_cancellations.get(job_id)
        if event is not None:
            event.set()
