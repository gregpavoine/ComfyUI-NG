# ComfyUI-NG

ComfyUI-NG is a typed, local-first control plane for modern image-generation
workflows. The core package targets Python 3.14 and intentionally excludes ML
runtimes and provider SDKs; those integrations remain isolated optional
components.

## Development installation

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The project exposes a `comfyng` command with the `serve`, `doctor`,
`benchmark`, `models`, `plugins`, `jobs`, `cache`, and `workers` surfaces.

## Configuration

The strict defaults live in `config/default.yaml`. Load them from Python with:

```python
from comfyng.config import load_settings

settings = load_settings()
```

Pass a YAML path to override defaults. Environment variables have final
precedence and use a double underscore for nesting, for example
`COMFYNG_SERVER__PORT=9000`. YAML values may reference required environment
variables as `${NAME}`.

Data defaults to `$COMFYNG_HOME`, then `$XDG_DATA_HOME/comfyui-ng`, then
`~/.local/share/comfyui-ng`. Database and storage paths are confined below
that root.

## Verification

```bash
python3.14 -m pytest tests/unit/test_config.py tests/architecture/test_core_imports.py -q
python3.14 -m build
comfyng --help
```
