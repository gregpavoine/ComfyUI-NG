from __future__ import annotations

import gc
import hashlib
import io
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers import FluxTransformer2DModel, FluxPipeline, AutoencoderKL

try:
    from accelerate import init_empty_weights
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    def init_empty_weights():
        class _EmptyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return _EmptyContext()


@dataclass(slots=True)
class ModelHandles:
    transformer: Optional[FluxTransformer2DModel] = None
    text_encoder: Optional[T5EncoderModel] = None
    tokenizer: Optional[T5TokenizerFast] = None
    vae: Optional[AutoencoderKL] = None
    pipeline: Optional[FluxPipeline] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    loaded: bool = False


class FluxRuntime:
    """FLUX.1 model runtime for ComfyUI-NG worker process."""

    def __init__(self) -> None:
        self._handles = ModelHandles()
        self._lock = threading.RLock()
        self._model_path: Optional[Path] = None
        self._cancellation: Optional[threading.Event] = None

    def _resolve_device(self) -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _load_transformer(self, model_path: Path, device: str, dtype: torch.dtype) -> FluxTransformer2DModel:
        transformer_path = model_path / "transformer"
        if transformer_path.exists():
            config_path = transformer_path / "config.json"
            if config_path.exists():
                return FluxTransformer2DModel.from_pretrained(
                    str(transformer_path),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(device)
        transformer_files = list(model_path.glob("*transformer*.safetensors"))
        if transformer_files:
            state_dict = load_file(str(transformer_files[0]), device="cpu")
            if not ACCELERATE_AVAILABLE:
                raise RuntimeError("accelerate required for single-file transformer loading")
            with init_empty_weights():
                model = FluxTransformer2DModel.from_config(
                    FluxTransformer2DModel.load_config(str(model_path / "transformer" / "config.json"))
                )
            model.load_state_dict(state_dict, strict=True, assign=True)
            return model.to(device=device, dtype=dtype)
        raise FileNotFoundError("FLUX transformer not found in model path")

    def _load_text_encoder(self, model_path: Path, device: str, dtype: torch.dtype) -> tuple[T5EncoderModel, T5TokenizerFast]:
        text_encoder_path = model_path / "text_encoder"
        if text_encoder_path.exists():
            tokenizer = T5TokenizerFast.from_pretrained(str(text_encoder_path))
            text_encoder = T5EncoderModel.from_pretrained(
                str(text_encoder_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            return text_encoder, tokenizer
        tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
        text_encoder = T5EncoderModel.from_pretrained(
            "google/t5-v1_1-xxl",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        return text_encoder, tokenizer

    def _load_vae(self, model_path: Path, device: str, dtype: torch.dtype) -> AutoencoderKL:
        vae_path = model_path / "vae"
        if vae_path.exists():
            return AutoencoderKL.from_pretrained(
                str(vae_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
        return AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)

    def execute(
        self,
        operation: str,
        payload: Mapping[str, Any],
        cancellation: threading.Event,
    ) -> Any:
        self._cancellation = cancellation

        if operation == "ng.model.flux.load":
            return self._load_model(payload)
        if operation == "ng.text_encoder.load":
            return self._load_text_encoder_op(payload)
        if operation == "ng.vae.load":
            return self._load_vae_op(payload)
        if operation == "ng.model.inspect":
            return self._inspect_model()
        if operation == "ng.model.unload":
            return self._unload_model()
        if operation == "ng.sample.flux":
            return self._sample_flux(payload)
        if operation == "ng.sample.flux_advanced":
            return self._sample_flux_advanced(payload)

        raise ValueError(f"unknown operation: {operation}")

    def _load_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        model_path = Path(payload.get("model_path", "")).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        device, dtype = self._resolve_device()

        with self._lock:
            if self._handles.loaded and self._model_path == model_path:
                return {"status": "already_loaded", "device": device, "dtype": str(dtype)}

            self._handles.device = device
            self._handles.dtype = dtype

            self._handles.transformer = self._load_transformer(model_path, device, dtype)
            self._handles.text_encoder, self._handles.tokenizer = self._load_text_encoder(model_path, device, dtype)
            self._handles.vae = self._load_vae(model_path, device, dtype)

            self._handles.pipeline = FluxPipeline(
                transformer=self._handles.transformer,
                text_encoder=self._handles.text_encoder,
                tokenizer=self._handles.tokenizer,
                vae=self._handles.vae,
            )
            self._handles.pipeline.to(device)

            self._handles.loaded = True
            self._model_path = model_path

            if hasattr(self._handles.pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self._handles.pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            if hasattr(torch, "compile") and payload.get("compile", False):
                self._handles.pipeline.transformer = torch.compile(
                    self._handles.pipeline.transformer,
                    mode="reduce-overhead",
                    fullgraph=True,
                )

        return {
            "status": "loaded",
            "device": device,
            "dtype": str(dtype),
            "model_path": str(model_path),
        }

    def _load_text_encoder_op(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        model_path = Path(payload.get("model_path", "")).expanduser().resolve()
        device, dtype = self._resolve_device()

        with self._lock:
            text_encoder, tokenizer = self._load_text_encoder(model_path, device, dtype)
            self._handles.text_encoder = text_encoder
            self._handles.tokenizer = tokenizer

        return {"status": "loaded", "device": device}

    def _load_vae_op(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        model_path = Path(payload.get("model_path", "")).expanduser().resolve()
        device, dtype = self._resolve_device()

        with self._lock:
            vae = self._load_vae(model_path, device, dtype)
            self._handles.vae = vae
            if self._handles.pipeline:
                self._handles.pipeline.vae = vae

        return {"status": "loaded", "device": device}

    def _inspect_model(self) -> dict[str, Any]:
        with self._lock:
            if not self._handles.loaded:
                return {"loaded": False}

            info = {
                "loaded": True,
                "device": self._handles.device,
                "dtype": str(self._handles.dtype),
                "model_path": str(self._model_path) if self._model_path else None,
                "components": {
                    "transformer": self._handles.transformer is not None,
                    "text_encoder": self._handles.text_encoder is not None,
                    "tokenizer": self._handles.tokenizer is not None,
                    "vae": self._handles.vae is not None,
                    "pipeline": self._handles.pipeline is not None,
                },
            }

            if self._handles.transformer:
                info["transformer_params"] = sum(
                    p.numel() for p in self._handles.transformer.parameters()
                )

            return info

    def _unload_model(self) -> dict[str, Any]:
        with self._lock:
            released = 0
            if self._handles.pipeline:
                self._handles.pipeline = None
                released += 1
            if self._handles.transformer:
                self._handles.transformer = None
                released += 1
            if self._handles.text_encoder:
                self._handles.text_encoder = None
                released += 1
            if self._handles.tokenizer:
                self._handles.tokenizer = None
            if self._handles.vae:
                self._handles.vae = None
                released += 1

            self._handles.loaded = False
            self._model_path = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return {"released": released}

    def _sample_flux(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self._handles.pipeline:
            raise RuntimeError("Model not loaded. Call ng.model.flux.load first.")

        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "")
        width = payload.get("width", 1024)
        height = payload.get("height", 1024)
        num_inference_steps = payload.get("steps", 28)
        guidance_scale = payload.get("guidance", 3.5)
        seed = payload.get("seed", 42)
        num_images = payload.get("num_images", 1)

        device = self._handles.device
        generator = torch.Generator(device=device).manual_seed(seed)

        with self._lock:
            if self._cancellation and self._cancellation.is_set():
                raise RuntimeError("Generation cancelled")

            with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], dtype=self._handles.dtype):
                images = self._handles.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=num_images,
                ).images

        results = []
        for i, img in enumerate(images):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            digest = hashlib.sha256(img_bytes).hexdigest()

            results.append({
                "index": i,
                "width": img.width,
                "height": img.height,
                "digest": f"sha256:{digest}",
                "bytes": img_bytes,
            })

        return {"images": results, "seed": seed}

    def _sample_flux_advanced(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._sample_flux(payload)

    def unload(self) -> Mapping[str, Any]:
        return self._unload_model()


def create_runtime() -> FluxRuntime:
    return FluxRuntime()