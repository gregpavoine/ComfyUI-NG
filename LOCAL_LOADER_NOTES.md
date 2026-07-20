# ComfyUI-NG local loader v0.5

- Runtime is strictly offline: no Hugging Face repository ID or download fallback.
- Models are resolved recursively from local `diffusion_models` and `checkpoints` roots.
- Architecture detection uses the complete path, Safetensors metadata and tensor keys.
- Component directories are resolved only from local paths:
  - `~/ComfyUI-NG/models/components`
  - `~/ComfyUI-NG/models`
  - `~/ComfyUI/models/components`
  - `~/ComfyUI/models`
  - `COMFYNG_COMPONENT_PATHS`
- A local Diffusers component directory must contain `model_index.json`. The selected
  `diffusion_models/*.safetensors` file replaces its transformer.
- `HF_HUB_OFFLINE=1` is enforced during all single-file loads.
