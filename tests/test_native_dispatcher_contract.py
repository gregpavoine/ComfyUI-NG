from __future__ import annotations

import asyncio
from pathlib import Path
import threading

from comfyng.api.dispatcher import WorkflowDispatcher


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def execute(self, operation, payload, cancellation: threading.Event):
        self.calls.append((operation, dict(payload)))
        if operation == "ng.model.flux.load":
            return {"status": "loaded", "model_source": payload["model_path"]}
        if operation == "ng.lora.load":
            return {"name": payload["adapter_name"], "weight": payload["model_strength"]}
        if operation == "ng.sample.flux":
            return {"seed": payload["seed"], "images": [{"bytes": b"PNG", "width": 1, "height": 1, "digest": "sha256:x"}]}
        raise AssertionError(operation)


def test_source_has_no_legacy_comfyui_bridge():
    source = Path("src/comfyng/api/dispatcher.py").read_text(encoding="utf-8")
    for forbidden in ("127.0.0.1:8188", "CheckpointLoaderSimple", "urllib.request", '"/prompt"'):
        assert forbidden not in source


def test_direct_native_runtime(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"lora")
    runtime = FakeRuntime()
    dispatcher = WorkflowDispatcher(tmp_path / "out", runtime=runtime)
    request = dispatcher._extract_request({"model_path": str(model), "prompt": "hello", "loras": [{"path": str(lora), "weight": 0.5}]})
    assert request["prompt"] == "hello"
    assert request["loras"][0]["weight"] == 0.5
    assert dispatcher._resolve_path(str(model), kind="model") == str(model.resolve())
