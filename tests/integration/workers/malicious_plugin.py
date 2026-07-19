"""Import-side-effect fixture used to prove sandbox ordering."""

from __future__ import annotations

import os
from pathlib import Path

from comfyng.runtime.entrypoint import DefaultRuntime


_PROBE_PATH = os.environ.get("COMFYNG_MALICIOUS_READ")
if _PROBE_PATH is not None:
    Path(_PROBE_PATH).read_text(encoding="utf-8")


def create_runtime() -> DefaultRuntime:
    return DefaultRuntime()
