"""ComfyUI-NG core package.

The package root stays deliberately dependency-light. Runtime and provider
integrations are imported only by their owning isolated processes.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
