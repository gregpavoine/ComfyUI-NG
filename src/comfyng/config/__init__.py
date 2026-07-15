"""Strict ComfyUI-NG configuration."""

from .loader import load_settings
from .models import Settings

__all__ = ["Settings", "load_settings"]
