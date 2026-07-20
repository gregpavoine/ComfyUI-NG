# ComfyUI-NG native recovery

This build removes every automatic HTTP call to legacy ComfyUI (`:8188`).

The generation path is now:

`API -> Scheduler -> WorkflowDispatcher -> native FLUX runtime -> artifact`

## Model requirement

The first native runtime accepts a complete local FLUX Diffusers repository containing `model_index.json`, or a Hugging Face repository id when remote loading is explicitly enabled later. A lone ComfyUI `diffusion_models/*.safetensors` transformer is deliberately rejected because it is not a complete pipeline.

Set an absolute model directory in the UI/API, or configure:

```bash
export COMFYNG_MODEL_PATHS=/path/to/diffusers/models
export COMFYNG_LORA_PATHS=/home/gp/ComfyUI/models/loras
```

No fake image fallback, Pollinations call, `/prompt` proxy, `CheckpointLoaderSimple`, or `KSampler` bridge remains in the execution path.
