"""Model metadata and capability contracts."""

from .capabilities import ModelCapabilities, ModelHandle
from .detector import ArchitectureDetection, ArchitectureDetector
from .inspection import ModelInspection, ModelInspector
from .legacy import UnsupportedModelGeneration, require_modern
from .registry import ModelFile, ModelRegistry, ModelSource, RegisteredModel

__all__ = [
    "ArchitectureDetection",
    "ArchitectureDetector",
    "ModelCapabilities",
    "ModelFile",
    "ModelHandle",
    "ModelInspection",
    "ModelInspector",
    "ModelRegistry",
    "ModelSource",
    "RegisteredModel",
    "UnsupportedModelGeneration",
    "require_modern",
]
