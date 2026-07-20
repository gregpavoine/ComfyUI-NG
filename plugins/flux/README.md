# ComfyUI-NG FLUX Runtime

This package provides the FLUX.1 model runtime for ComfyUI-NG. It runs in an isolated GPU worker process and provides the following operations:

## Operations

| Operation | Description |
|-----------|-------------|
| `ng.model.flux.load` | Load full FLUX model (transformer, text encoder, VAE) |
| `ng.text_encoder.load` | Load only T5 text encoder |
| `ng.vae.load` | Load only VAE |
| `ng.model.inspect` | Inspect loaded model state |
| `ng.model.unload` | Unload all model components |
| `ng.sample.flux` | Generate images with FLUX |
| `ng.sample.flux_advanced` | Advanced sampling with more control |

## Model Directory Structure

The runtime expects the model path to contain:

```
model_path/
├── transformer/
│   ├── config.json
│   └── *.safetensors (or model.safetensors)
├── text_encoder/
│   ├── config.json
│   ├── tokenizer.json
│   └── *.safetensors
└── vae/
    ├── config.json
    └── *.safetensors
```

Or as single-file safetensors in the root directory.

## Installation

```bash
pip install -e runtimes/flux
```

Requires:
- Python 3.14+
- PyTorch 2.5+ with CUDA
- transformers 4.48+
- diffusers 0.33+
- accelerate 1.3+
- safetensors 0.5+

## Usage

The runtime is automatically loaded by the ComfyUI-NG worker supervisor when a plugin using this package is started. It communicates via the worker protocol.

## Example Worker Spec

```json
{
  "worker_id": "flux-worker-1",
  "kind": "GPU_MODEL",
  "entrypoint": "comfyng_flux.runtime:create_runtime",
  "sandbox": {
    "allow_network": false,
    "filesystem_read_roots": ["/models"],
    "filesystem_write_roots": ["/outputs"]
  }
}
```

## Nodes Exposed

- **FLUX Model Loader** (`ng.model.flux.load`) - Load complete model
- **FLUX Text Encoder Loader** (`ng.text_encoder.load`) - Load text encoder
- **FLUX VAE Loader** (`ng.vae.load`) - Load VAE
- **FLUX Model Inspector** (`ng.model.inspect`) - Check model status
- **FLUX Model Unload** (`ng.model.unload`) - Release VRAM
- **FLUX Sampler** (`ng.sample.flux`) - Generate images
- **FLUX Sampler (Advanced)** (`ng.sample.flux_advanced`) - Advanced generation