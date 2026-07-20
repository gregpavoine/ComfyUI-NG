from __future__ import annotations

import gc
import hashlib
import io
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers import FluxPipeline


class FluxRuntimeError(RuntimeError):
    pass


@dataclass(slots=True)
class LoraAdapter:
    name: str
    path: str
    weight: float
    sha256: str


@dataclass(slots=True)
class RuntimeState:
    pipeline: FluxPipeline | None = None
    model_source: str | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    adapters: dict[str, LoraAdapter] = field(default_factory=dict)


class FluxRuntime:
    """A minimal, honest FLUX runtime.

    Supported model inputs:
    - a local Diffusers FLUX directory containing model_index.json;
    - a Hugging Face repository id, only when local_files_only=False.

    Deliberately unsupported here:
    - a lone ComfyUI transformer/checkpoint safetensors file without its text
      encoders, tokenizer, scheduler and VAE configuration.
    """

    def __init__(self) -> None:
        self._state = RuntimeState()
        self._lock = threading.RLock()
        self._cancellation: threading.Event | None = None

    @staticmethod
    def _resolve_device(requested: str | None = None) -> tuple[str, torch.dtype]:
        if requested and requested != "auto":
            if requested.startswith("cuda") and not torch.cuda.is_available():
                raise FluxRuntimeError("CUDA was requested but is unavailable")
            if requested == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                raise FluxRuntimeError("MPS was requested but is unavailable")
            if requested.startswith("cuda"):
                return requested, torch.bfloat16
            if requested == "mps":
                return requested, torch.float16
            return requested, torch.float32
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def execute(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        self._cancellation = cancellation
        handlers = {
            "ng.model.flux.load": self._load_model,
            "ng.model.load": self._load_model,
            "ng.model.inspect": self._inspect_model,
            "ng.model.unload": self._unload_model,
            "ng.lora.load": self._load_lora,
            "ng.lora.stack": self._set_lora_stack,
            "ng.lora.inspect": self._inspect_loras,
            "ng.lora.unload": self._unload_loras,
            "ng.sample.flux": self._sample_flux,
            "ng.sample.run": self._sample_flux,
            "ng.sample.flux_advanced": self._sample_flux,
            "ng.sample.advanced": self._sample_flux,
        }
        try:
            handler = handlers[operation]
        except KeyError as exc:
            raise FluxRuntimeError(f"unknown FLUX runtime operation: {operation}") from exc
        return handler(payload)

    def _load_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source = str(payload.get("model_path") or payload.get("model_id") or "").strip()
        if not source:
            raise FluxRuntimeError("model_path or model_id is required")

        local_only = bool(payload.get("local_files_only", True))
        path = Path(source).expanduser()
        if path.exists():
            path = path.resolve()
            if path.is_file():
                raise FluxRuntimeError(
                    "A standalone safetensors checkpoint is not a complete FLUX pipeline. "
                    "Provide a Diffusers directory containing model_index.json and all required components."
                )
            if not (path / "model_index.json").is_file():
                raise FluxRuntimeError(
                    f"{path} is not a Diffusers pipeline directory: model_index.json is missing"
                )
            source = str(path)
        elif local_only:
            raise FileNotFoundError(
                f"Local model does not exist: {source}. Set local_files_only=false only for an explicit remote load."
            )

        device, dtype = self._resolve_device(str(payload.get("device", "auto")))
        with self._lock:
            if self._state.pipeline is not None and self._state.model_source == source:
                return self._status("already_loaded")

            self._unload_model({})
            kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "local_files_only": local_only,
            }
            pipeline = FluxPipeline.from_pretrained(source, **kwargs)

            memory_mode = str(payload.get("memory_mode", "auto"))
            if device.startswith("cuda") and memory_mode in {"offload", "low_memory"}:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(device)

            if bool(payload.get("vae_tiling", False)) and hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
            if bool(payload.get("vae_slicing", False)) and hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()

            compile_mode = payload.get("compile_mode")
            if compile_mode and compile_mode != "off":
                pipeline.transformer = torch.compile(
                    pipeline.transformer,
                    mode=str(compile_mode),
                    fullgraph=False,
                )

            self._state = RuntimeState(
                pipeline=pipeline,
                model_source=source,
                device=device,
                dtype=dtype,
            )
            return self._status("loaded")

    def _require_pipeline(self) -> FluxPipeline:
        pipeline = self._state.pipeline
        if pipeline is None:
            raise FluxRuntimeError("No FLUX pipeline is loaded")
        return pipeline

    def _load_lora(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self._require_pipeline()
        raw_path = str(payload.get("path") or "").strip()
        if not raw_path:
            raise FluxRuntimeError("LoRA path is required")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file does not exist: {path}")
        if path.suffix.lower() != ".safetensors":
            raise FluxRuntimeError("Only .safetensors LoRAs are accepted")

        name = str(payload.get("adapter_name") or path.stem).strip()
        weight = float(payload.get("model_strength", payload.get("weight", 1.0)))
        if not -4.0 <= weight <= 4.0:
            raise FluxRuntimeError("LoRA weight must be between -4 and 4")

        with self._lock:
            if name in self._state.adapters:
                raise FluxRuntimeError(f"LoRA adapter name already loaded: {name}")
            pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=name,
            )
            self._state.adapters[name] = LoraAdapter(
                name=name,
                path=str(path),
                weight=weight,
                sha256=self._sha256(path),
            )
            self._apply_adapter_stack()
        return self._inspect_loras({})

    def _set_lora_stack(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_pipeline()
        requested = payload.get("adapters")
        if not isinstance(requested, list) or not requested:
            raise FluxRuntimeError("adapters must be a non-empty list")
        with self._lock:
            for item in requested:
                if not isinstance(item, Mapping):
                    raise FluxRuntimeError("each adapter must be an object")
                name = str(item.get("name") or "")
                if name not in self._state.adapters:
                    raise FluxRuntimeError(f"unknown loaded adapter: {name}")
                self._state.adapters[name].weight = float(item.get("weight", 1.0))
            ordered = {str(item["name"]): self._state.adapters[str(item["name"])] for item in requested}
            self._state.adapters = ordered
            self._apply_adapter_stack()
        return self._inspect_loras({})

    def _apply_adapter_stack(self) -> None:
        pipeline = self._require_pipeline()
        names = list(self._state.adapters)
        if names:
            weights = [self._state.adapters[name].weight for name in names]
            pipeline.set_adapters(names, adapter_weights=weights)

    def _inspect_loras(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "loaded": [
                {
                    "name": adapter.name,
                    "path": adapter.path,
                    "weight": adapter.weight,
                    "sha256": adapter.sha256,
                }
                for adapter in self._state.adapters.values()
            ]
        }

    def _unload_loras(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self._require_pipeline()
        with self._lock:
            if self._state.adapters:
                try:
                    pipeline.unload_lora_weights(reset_to_overwritten_params=True)
                except TypeError:
                    pipeline.unload_lora_weights()
            count = len(self._state.adapters)
            self._state.adapters.clear()
        return {"released": count}

    def _sample_flux(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self._require_pipeline()
        if self._cancellation is not None and self._cancellation.is_set():
            raise FluxRuntimeError("Generation cancelled")

        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise FluxRuntimeError("prompt is required")
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        steps = int(payload.get("steps", 28))
        guidance = float(payload.get("guidance", 3.5))
        seed = int(payload.get("seed", 42))
        num_images = int(payload.get("num_images", 1))
        if width % 16 or height % 16:
            raise FluxRuntimeError("width and height must be multiples of 16")
        if not 1 <= steps <= 200:
            raise FluxRuntimeError("steps must be between 1 and 200")

        generator_device = "cuda" if self._state.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        callback = None
        callback_inputs = None
        if self._cancellation is not None:
            def callback(pipe: FluxPipeline, step: int, timestep: Any, callback_kwargs: dict[str, Any]):
                if self._cancellation is not None and self._cancellation.is_set():
                    raise FluxRuntimeError("Generation cancelled")
                return callback_kwargs
            callback_inputs = ["latents"]

        with self._lock, torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                num_images_per_prompt=num_images,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=callback_inputs,
            )

        images: list[dict[str, Any]] = []
        for index, image in enumerate(result.images):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data = buffer.getvalue()
            images.append(
                {
                    "index": index,
                    "width": image.width,
                    "height": image.height,
                    "digest": f"sha256:{hashlib.sha256(data).hexdigest()}",
                    "bytes": data,
                }
            )
        return {"images": images, "seed": seed, "loras": self._inspect_loras({})["loaded"]}

    def _inspect_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._status("loaded" if self._state.pipeline is not None else "unloaded")

    def _status(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "loaded": self._state.pipeline is not None,
            "model_source": self._state.model_source,
            "device": self._state.device,
            "dtype": str(self._state.dtype),
            "lora_count": len(self._state.adapters),
        }

    def _unload_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            pipeline = self._state.pipeline
            released = int(pipeline is not None)
            self._state = RuntimeState()
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        return {"released": released}

    def unload(self) -> Mapping[str, Any]:
        return self._unload_model({})


def create_runtime() -> FluxRuntime:
    return FluxRuntime()
