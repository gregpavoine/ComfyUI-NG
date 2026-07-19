"""Frozen graph contracts, validation and deterministic compilation."""

from importlib import import_module
from typing import Any

from .types import (
    DEFAULT_TYPE_REGISTRY,
    Edge,
    Graph,
    InputBinding,
    NodeInstance,
    OutputBinding,
    PortTypeDefinition,
    TensorHandle,
    TypeRef,
    TypeRegistry,
)


_LAZY_EXPORTS = {
    "CompileContext": (".compiler", "CompileContext"),
    "ControlAnnotation": (".compiler", "ControlAnnotation"),
    "ControlBranch": (".compiler", "ControlBranch"),
    "ControlRegion": (".compiler", "ControlRegion"),
    "CycleError": (".topology", "CycleError"),
    "ExecutionGroup": (".compiler", "ExecutionGroup"),
    "ExecutionPlan": (".compiler", "ExecutionPlan"),
    "ExecutionStep": (".compiler", "ExecutionStep"),
    "FusionEvidence": (".compiler", "FusionEvidence"),
    "GraphCompilationError": (".compiler", "GraphCompilationError"),
    "GraphCompiler": (".compiler", "GraphCompiler"),
    "GraphDiagnostic": (".validation", "GraphDiagnostic"),
    "ResourceAnnotation": (".compiler", "ResourceAnnotation"),
    "SubgraphExpansionError": (".subgraphs", "SubgraphExpansionError"),
    "SubgraphTrace": (".subgraphs", "SubgraphTrace"),
    "canonical_json": (".cache", "canonical_json"),
    "content_cache_key": (".cache", "content_cache_key"),
    "expand_subgraphs": (".subgraphs", "expand_subgraphs"),
    "expand_subgraphs_with_trace": (".subgraphs", "expand_subgraphs_with_trace"),
    "node_cache_key": (".cache", "node_cache_key"),
    "topological_layers": (".topology", "topological_layers"),
    "topological_sort": (".topology", "topological_sort"),
    "validate_graph": (".validation", "validate_graph"),
    "validate_graph_structure": (".validation", "validate_graph_structure"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "CompileContext",
    "ControlAnnotation",
    "ControlBranch",
    "ControlRegion",
    "CycleError",
    "DEFAULT_TYPE_REGISTRY",
    "Edge",
    "ExecutionGroup",
    "ExecutionPlan",
    "ExecutionStep",
    "FusionEvidence",
    "Graph",
    "GraphCompilationError",
    "GraphCompiler",
    "GraphDiagnostic",
    "InputBinding",
    "NodeInstance",
    "OutputBinding",
    "PortTypeDefinition",
    "ResourceAnnotation",
    "SubgraphExpansionError",
    "SubgraphTrace",
    "TensorHandle",
    "TypeRef",
    "TypeRegistry",
    "canonical_json",
    "content_cache_key",
    "expand_subgraphs",
    "expand_subgraphs_with_trace",
    "node_cache_key",
    "topological_layers",
    "topological_sort",
    "validate_graph",
    "validate_graph_structure",
]
