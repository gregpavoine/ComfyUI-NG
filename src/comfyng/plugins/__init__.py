"""Declarative plugin manifest and node catalogue APIs."""

from .catalogue import NodeCatalogue
from .manifest import (
    NodeDefinition,
    NodeExecutionTraits,
    PackageMetadata,
    PluginManifest,
    ResourceRequirements,
    RuntimeDefinition,
    load_json_schema,
)

__all__ = [
    "NodeCatalogue",
    "NodeDefinition",
    "NodeExecutionTraits",
    "PackageMetadata",
    "PluginManifest",
    "ResourceRequirements",
    "RuntimeDefinition",
    "load_json_schema",
]
