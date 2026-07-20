# Native diffusion_models loader

The selected file under `models/diffusion_models` is loaded as the transformer.
ComfyUI-NG then hydrates tokenizer(s), text encoder(s), VAE and scheduler from a
local Diffusers component bundle.

Supported family selectors:

- `flux`
- `flux_schnell`
- `z_image`
- `z_image_turbo`
- `krea2`
- `krea2_turbo`

Component bundle environment variables:

```bash
export COMFYNG_FLUX_COMPONENTS=/path/to/flux-dev-components
export COMFYNG_FLUX_SCHNELL_COMPONENTS=/path/to/flux-schnell-components
export COMFYNG_ZIMAGE_COMPONENTS=/path/to/z-image-components
export COMFYNG_ZIMAGE_TURBO_COMPONENTS=/path/to/z-image-turbo-components
export COMFYNG_KREA2_COMPONENTS=/path/to/krea2-diffusers
```

Each component bundle must contain `model_index.json` and all non-transformer
components required by its pipeline. The transformer's weights in that bundle
are ignored and replaced by the selected `diffusion_models/*.safetensors` file.

For FLUX and Z-Image, a cached Hugging Face snapshot is detected automatically.
Remote hydration occurs only when `local_files_only=false` is explicitly used.
Krea 2 currently requires a local converted Diffusers bundle.
