from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import torch


class NativeImageRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArchitectureProfile:
    family: str
    pipeline_class: str
    transformer_class: str
    default_steps: int
    default_guidance: float
    component_env: str
    filename_markers: tuple[str, ...]
    tensor_markers: tuple[str, ...]
    distilled: bool = False


PROFILES: tuple[ArchitectureProfile, ...] = (
    ArchitectureProfile(
        family="krea2_turbo",
        pipeline_class="Krea2Pipeline",
        transformer_class="Krea2Transformer2DModel",
        default_steps=8,
        default_guidance=0.0,
        component_env="COMFYNG_KREA2_COMPONENTS",
        filename_markers=("krea2", "krea_2", "k2", "tdm", "turbo"),
        tensor_markers=("text_fuser", "text_fusion", "num_text_layers"),
        distilled=True,
    ),
    ArchitectureProfile(
        family="krea2",
        pipeline_class="Krea2Pipeline",
        transformer_class="Krea2Transformer2DModel",
        default_steps=28,
        default_guidance=4.5,
        component_env="COMFYNG_KREA2_COMPONENTS",
        filename_markers=("krea2", "krea_2", "k2"),
        tensor_markers=("text_fuser", "text_fusion", "num_text_layers"),
    ),
    ArchitectureProfile(
        family="z_image_turbo",
        pipeline_class="ZImagePipeline",
        transformer_class="ZImageTransformer2DModel",
        default_steps=8,
        default_guidance=0.0,
        component_env="COMFYNG_ZIMAGE_TURBO_COMPONENTS",
        filename_markers=("zimage", "z-image", "z_image", "zpop", "turbo"),
        tensor_markers=("cap_embedder", "x_embedder", "noise_refiner"),
        distilled=True,
    ),
    ArchitectureProfile(
        family="z_image",
        pipeline_class="ZImagePipeline",
        transformer_class="ZImageTransformer2DModel",
        default_steps=50,
        default_guidance=5.0,
        component_env="COMFYNG_ZIMAGE_COMPONENTS",
        filename_markers=("zimage", "z-image", "z_image", "zpop"),
        tensor_markers=("cap_embedder", "x_embedder", "noise_refiner"),
    ),
    ArchitectureProfile(
        family="flux_schnell",
        pipeline_class="FluxPipeline",
        transformer_class="FluxTransformer2DModel",
        default_steps=4,
        default_guidance=0.0,
        component_env="COMFYNG_FLUX_SCHNELL_COMPONENTS",
        filename_markers=("flux", "schnell"),
        tensor_markers=("double_blocks", "single_blocks", "img_in"),
        distilled=True,
    ),
    ArchitectureProfile(
        family="flux",
        pipeline_class="FluxPipeline",
        transformer_class="FluxTransformer2DModel",
        default_steps=28,
        default_guidance=3.5,
        component_env="COMFYNG_FLUX_COMPONENTS",
        filename_markers=("flux",),
        tensor_markers=("double_blocks", "single_blocks", "img_in"),
    ),
)


@dataclass(slots=True)
class LoraAdapter:
    name: str
    path: str
    weight: float
    sha256: str


@dataclass(slots=True)
class RuntimeState:
    pipeline: Any | None = None
    model_source: str | None = None
    component_source: str | None = None
    family: str | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    default_steps: int = 28
    default_guidance: float = 3.5
    adapters: dict[str, LoraAdapter] = field(default_factory=dict)


class ModernImageRuntime:
    """Native loader for ComfyUI-style diffusion transformer files.

    A file in models/diffusion_models is treated as the denoising transformer.
    The remaining pipeline components are hydrated from a local Diffusers bundle
    or, only when explicitly allowed, from a Hugging Face repository/cache.
    """

    def __init__(self) -> None:
        self._state = RuntimeState()
        self._lock = threading.RLock()
        self._cancellation: threading.Event | None = None

    @staticmethod
    def _resolve_device(requested: str | None = None) -> tuple[str, torch.dtype]:
        requested = requested or "auto"
        if requested != "auto":
            if requested.startswith("cuda") and not torch.cuda.is_available():
                raise NativeImageRuntimeError("CUDA was requested but is unavailable")
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

    @staticmethod
    def _safetensor_keys(path: Path, limit: int = 512) -> tuple[str, ...]:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise NativeImageRuntimeError(
                "safetensors is required to inspect diffusion_models checkpoints"
            ) from exc
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                return tuple(list(handle.keys())[:limit])
        except Exception as exc:
            raise NativeImageRuntimeError(f"Unable to inspect safetensors file {path}: {exc}") from exc

    @classmethod
    def _detect_profile(cls, path: Path, requested_family: str | None = None) -> ArchitectureProfile:
        if requested_family and requested_family != "auto":
            normalized = requested_family.lower().replace("-", "_")
            for profile in PROFILES:
                if profile.family == normalized:
                    return profile
            raise NativeImageRuntimeError(
                f"Unknown model family {requested_family!r}. Supported: "
                + ", ".join(profile.family for profile in PROFILES)
            )

        # Use the complete path: users commonly sort models under folders such as
        # Krea 2/, ZImageTurbo/ and Flux/. The former implementation only inspected
        # path.name and therefore missed that reliable local information.
        path_blob = str(path).lower().replace("-", "_").replace(" ", "_")
        keys = cls._safetensor_keys(path, limit=4096)
        key_blob = "\n".join(keys).lower()

        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
        except Exception:
            metadata = {}
        metadata_blob = json.dumps(metadata, ensure_ascii=False).lower()
        evidence = "\n".join((path_blob, key_blob, metadata_blob))

        if any(x in evidence for x in ("krea2", "krea_2", "krea-2")):
            return PROFILES[0] if any(x in evidence for x in ("turbo", "tdm", "distill")) else PROFILES[1]
        if any(x in evidence for x in ("zimage", "z_image", "z-image", "z_image_turbo")):
            return PROFILES[2] if any(x in evidence for x in ("turbo", "distill")) else PROFILES[3]
        if "flux" in evidence or sum(x in key_blob for x in ("double_blocks", "single_blocks", "img_in")) >= 2:
            return PROFILES[4] if any(x in evidence for x in ("schnell", "turbo", "distill")) else PROFILES[5]

        # Architecture-specific tensor signatures. Krea2 and Z-Image community
        # checkpoints often omit metadata, so use multiple independent markers.
        if sum(x in key_blob for x in ("text_fuser", "text_fusion", "num_text_layers")) >= 2:
            return PROFILES[0] if any(x in evidence for x in ("turbo", "tdm", "distill")) else PROFILES[1]
        if sum(x in key_blob for x in ("cap_embedder", "context_refiner", "noise_refiner", "x_embedder")) >= 2:
            return PROFILES[2] if any(x in evidence for x in ("turbo", "distill")) else PROFILES[3]

        raise NativeImageRuntimeError(
            f"Unable to detect architecture for local model {path}. "
            "Select model_family explicitly in the workflow. No network fallback is permitted."
        )

    @staticmethod
    def _component_roots() -> tuple[Path, ...]:
        roots: list[Path] = []
        for value in os.environ.get("COMFYNG_COMPONENT_PATHS", "").split(os.pathsep):
            if value:
                roots.append(Path(value).expanduser())
        home = Path.home()
        roots.extend((
            home / "ComfyUI-NG" / "models" / "components",
            home / "ComfyUI-NG" / "models",
            home / "ComfyUI" / "models" / "components",
            home / "ComfyUI" / "models",
        ))
        return tuple(dict.fromkeys(path.resolve() for path in roots if path.exists()))

    @classmethod
    def _resolve_component_source(
        cls,
        profile: ArchitectureProfile,
        payload: Mapping[str, Any],
        *,
        local_files_only: bool = True,
    ) -> str:
        # Strictly local by design. Repository identifiers and cache downloads are
        # deliberately rejected: NG must consume the user's existing model tree.
        explicit = str(payload.get("component_source") or payload.get("components_path") or "").strip()
        env_value = os.environ.get(profile.component_env, "").strip()
        source = explicit or env_value
        if source:
            path = Path(source).expanduser()
            if not path.exists():
                raise NativeImageRuntimeError(f"Local component path does not exist: {path}")
            path = path.resolve()
            if not (path / "model_index.json").is_file():
                raise NativeImageRuntimeError(f"Local component directory {path} has no model_index.json")
            return str(path)

        expected = profile.pipeline_class.lower()
        family_tokens = tuple(token for token in profile.family.split("_") if token)
        candidates: list[tuple[int, Path]] = []
        for root in cls._component_roots():
            try:
                index_files = root.rglob("model_index.json")
                for index_file in index_files:
                    try:
                        cfg = json.loads(index_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    class_name = str(cfg.get("_class_name") or "").lower()
                    if class_name != expected:
                        continue
                    blob = str(index_file.parent).lower().replace("-", "_")
                    score = sum(token in blob for token in family_tokens)
                    if profile.distilled and any(x in blob for x in ("turbo", "schnell", "distill")):
                        score += 3
                    candidates.append((score, index_file.parent.resolve()))
            except (OSError, PermissionError):
                continue
        if candidates:
            candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
            return str(candidates[0][1])

        searched = ", ".join(str(p) for p in cls._component_roots()) or "(none)"
        raise NativeImageRuntimeError(
            f"Local components for {profile.family} were not found. Searched: {searched}. "
            f"Place a local {profile.pipeline_class} component directory under "
            "ComfyUI-NG/models/components, or set " + profile.component_env + ". "
            "Network downloads are disabled."
        )

    @staticmethod
    def _import_diffusers_class(name: str) -> type[Any]:
        try:
            module = import_module("diffusers")
            value = getattr(module, name)
        except (ImportError, AttributeError) as exc:
            raise NativeImageRuntimeError(
                f"Installed diffusers does not provide {name}. Upgrade diffusers to a version "
                "supporting this architecture."
            ) from exc
        return value

    @staticmethod
    def _from_single_file(
        transformer_class: type[Any],
        model_path: Path,
        component_source: str,
        dtype: torch.dtype,
        local_files_only: bool = True,
    ) -> Any:
        loader = getattr(transformer_class, "from_single_file", None)
        if not callable(loader):
            raise NativeImageRuntimeError(
                f"{transformer_class.__name__} does not support from_single_file()"
            )

        component_dir = Path(component_source)
        local_configs = [
            model_path.with_suffix(".json"),
            model_path.parent / "config.json",
            component_dir / "transformer" / "config.json",
            component_dir / "config.json",
        ]
        attempts: list[dict[str, Any]] = [{}]
        for config in local_configs:
            if config.is_file():
                attempts.append({"config": str(config)})
        if (component_dir / "transformer").is_dir():
            attempts.append({"config": str(component_dir), "subfolder": "transformer"})

        errors: list[str] = []
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            for extra in attempts:
                try:
                    return loader(
                        str(model_path),
                        torch_dtype=dtype,
                        local_files_only=True,
                        **extra,
                    )
                except Exception as exc:
                    errors.append(f"{extra or {'config': 'embedded'}}: {type(exc).__name__}: {exc}")
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        raise NativeImageRuntimeError(
            f"Failed to load local transformer {model_path.name} as {transformer_class.__name__}. "
            "No network request was attempted. Local attempts: " + " | ".join(errors)
        )

    def execute(self, operation: str, payload: Mapping[str, Any], cancellation: threading.Event) -> Any:
        self._cancellation = cancellation
        handlers = {
            "ng.model.load": self._load_model,
            "ng.model.flux.load": self._load_model,
            "ng.model.inspect": self._inspect_model,
            "ng.model.unload": self._unload_model,
            "ng.lora.load": self._load_lora,
            "ng.lora.stack": self._set_lora_stack,
            "ng.lora.inspect": self._inspect_loras,
            "ng.lora.unload": self._unload_loras,
            "ng.sample.run": self._sample,
            "ng.sample.flux": self._sample,
            "ng.sample.advanced": self._sample,
        }
        try:
            return handlers[operation](payload)
        except KeyError as exc:
            raise NativeImageRuntimeError(f"Unknown runtime operation: {operation}") from exc

    def _load_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_source = str(payload.get("model_path") or payload.get("model_id") or "").strip()
        if not raw_source:
            raise NativeImageRuntimeError("model_path is required")
        path = Path(raw_source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Diffusion model does not exist: {path}")

        local_files_only = True
        device, dtype = self._resolve_device(str(payload.get("device", "auto")))

        if path.is_dir():
            if not (path / "model_index.json").is_file():
                raise NativeImageRuntimeError(f"Pipeline directory {path} has no model_index.json")
            # Full pipeline directories remain supported.
            config = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
            pipeline_name = str(config.get("_class_name") or "FluxPipeline")
            pipeline_class = self._import_diffusers_class(pipeline_name)
            pipeline = pipeline_class.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
            family = pipeline_name.removesuffix("Pipeline").lower()
            component_source = str(path)
            default_steps = int(payload.get("steps") or 28)
            default_guidance = float(payload.get("guidance") or 3.5)
        else:
            if path.suffix.lower() not in {".safetensors", ".gguf"}:
                raise NativeImageRuntimeError("diffusion_models loader accepts .safetensors or .gguf")
            if path.suffix.lower() == ".gguf":
                raise NativeImageRuntimeError(
                    "GGUF transformer loading requires an explicit quantization backend and is not silently treated as safetensors"
                )
            profile = self._detect_profile(path, str(payload.get("model_family") or "auto"))
            component_source = self._resolve_component_source(
                profile, payload, local_files_only=local_files_only
            )
            transformer_class = self._import_diffusers_class(profile.transformer_class)
            pipeline_class = self._import_diffusers_class(profile.pipeline_class)
            transformer = self._from_single_file(
                transformer_class,
                path,
                component_source,
                dtype,
                local_files_only,
            )
            kwargs: dict[str, Any] = {
                "transformer": transformer,
                "torch_dtype": dtype,
                "local_files_only": True,
            }
            if profile.family.startswith("krea2"):
                kwargs["is_distilled"] = profile.distilled
            pipeline = pipeline_class.from_pretrained(component_source, **kwargs)
            family = profile.family
            default_steps = profile.default_steps
            default_guidance = profile.default_guidance

        memory_mode = str(payload.get("memory_mode") or ("offload" if payload.get("cpu_offload", True) else "normal"))
        if device.startswith("cuda") and memory_mode in {"offload", "low_memory", "auto"}:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)

        if bool(payload.get("vae_tiling", True)) and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if bool(payload.get("vae_slicing", True)) and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()

        compile_mode = str(payload.get("compile_mode") or "off")
        if compile_mode != "off" and hasattr(pipeline, "transformer"):
            pipeline.transformer = torch.compile(pipeline.transformer, mode=compile_mode, fullgraph=False)

        with self._lock:
            self._unload_model({})
            self._state = RuntimeState(
                pipeline=pipeline,
                model_source=str(path),
                component_source=component_source,
                family=family,
                device=device,
                dtype=dtype,
                default_steps=default_steps,
                default_guidance=default_guidance,
            )
        return self._status("loaded")

    def _require_pipeline(self) -> Any:
        if self._state.pipeline is None:
            raise NativeImageRuntimeError("No image pipeline is loaded")
        return self._state.pipeline

    def _load_lora(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self._require_pipeline()
        path = Path(str(payload.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file does not exist: {path}")
        if path.suffix.lower() != ".safetensors":
            raise NativeImageRuntimeError("Only .safetensors LoRAs are accepted")
        name = str(payload.get("adapter_name") or path.stem).strip()
        weight = float(payload.get("model_strength", payload.get("weight", 1.0)))
        if not -4.0 <= weight <= 4.0:
            raise NativeImageRuntimeError("LoRA weight must be between -4 and 4")
        if not callable(getattr(pipeline, "load_lora_weights", None)):
            raise NativeImageRuntimeError(f"{type(pipeline).__name__} does not expose a LoRA loader")
        with self._lock:
            if name in self._state.adapters:
                raise NativeImageRuntimeError(f"LoRA adapter already loaded: {name}")
            pipeline.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)
            self._state.adapters[name] = LoraAdapter(name, str(path), weight, self._sha256(path))
            self._apply_adapter_stack()
        return self._inspect_loras({})

    def _set_lora_stack(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        requested = payload.get("adapters")
        if not isinstance(requested, list) or not requested:
            raise NativeImageRuntimeError("adapters must be a non-empty list")
        with self._lock:
            ordered: dict[str, LoraAdapter] = {}
            for item in requested:
                if not isinstance(item, Mapping):
                    raise NativeImageRuntimeError("each adapter must be an object")
                name = str(item.get("name") or "")
                if name not in self._state.adapters:
                    raise NativeImageRuntimeError(f"unknown loaded adapter: {name}")
                adapter = self._state.adapters[name]
                adapter.weight = float(item.get("weight", adapter.weight))
                ordered[name] = adapter
            self._state.adapters = ordered
            self._apply_adapter_stack()
        return self._inspect_loras({})

    def _apply_adapter_stack(self) -> None:
        pipeline = self._require_pipeline()
        names = list(self._state.adapters)
        if names:
            setter = getattr(pipeline, "set_adapters", None)
            if not callable(setter):
                raise NativeImageRuntimeError(f"{type(pipeline).__name__} cannot stack LoRAs")
            setter(names, adapter_weights=[self._state.adapters[name].weight for name in names])

    def _inspect_loras(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {"loaded": [adapter.__dict__ if hasattr(adapter, "__dict__") else {
            "name": adapter.name, "path": adapter.path, "weight": adapter.weight, "sha256": adapter.sha256
        } for adapter in self._state.adapters.values()]}

    def _unload_loras(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self._require_pipeline()
        count = len(self._state.adapters)
        if count and callable(getattr(pipeline, "unload_lora_weights", None)):
            try:
                pipeline.unload_lora_weights(reset_to_overwritten_params=True)
            except TypeError:
                pipeline.unload_lora_weights()
        self._state.adapters.clear()
        return {"released": count}

    def _sample(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pipeline = self._require_pipeline()
        if self._cancellation is not None and self._cancellation.is_set():
            raise NativeImageRuntimeError("Generation cancelled")
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise NativeImageRuntimeError("prompt is required")
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        steps = int(payload.get("steps") or self._state.default_steps)
        guidance = float(payload.get("guidance") if payload.get("guidance") is not None else self._state.default_guidance)
        seed = int(payload.get("seed", 42))
        num_images = int(payload.get("num_images", 1))
        generator_device = "cuda" if self._state.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        def callback(*args: Any, **kwargs: Any) -> Any:
            if self._cancellation is not None and self._cancellation.is_set():
                raise NativeImageRuntimeError("Generation cancelled")
            if args and isinstance(args[-1], dict):
                return args[-1]
            return kwargs.get("callback_kwargs", {})

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "num_images_per_prompt": num_images,
            "callback_on_step_end": callback,
        }
        negative_prompt = str(payload.get("negative_prompt") or "")
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        with self._lock, torch.inference_mode():
            result = pipeline(**call_kwargs)

        images: list[dict[str, Any]] = []
        for index, image in enumerate(result.images):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            data = buffer.getvalue()
            images.append({
                "index": index,
                "width": image.width,
                "height": image.height,
                "digest": f"sha256:{hashlib.sha256(data).hexdigest()}",
                "bytes": data,
            })
        return {"images": images, "seed": seed, "family": self._state.family, "loras": self._inspect_loras({})["loaded"]}

    def _inspect_model(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._status("loaded" if self._state.pipeline is not None else "unloaded")

    def _status(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "loaded": self._state.pipeline is not None,
            "family": self._state.family,
            "model_source": self._state.model_source,
            "component_source": self._state.component_source,
            "device": self._state.device,
            "dtype": str(self._state.dtype),
            "default_steps": self._state.default_steps,
            "default_guidance": self._state.default_guidance,
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


# Backward-compatible factory name expected by the current plugin manifest.
FluxRuntime = ModernImageRuntime


def create_runtime() -> ModernImageRuntime:
    return ModernImageRuntime()
