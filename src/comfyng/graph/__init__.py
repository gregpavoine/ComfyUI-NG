"""Frozen graph contracts, validation and deterministic compilation."""

from .cache import canonical_json, content_cache_key, node_cache_key
from .compiler import (
    CompileContext,
    ControlAnnotation,
    ExecutionGroup,
    ExecutionPlan,
    ExecutionStep,
    GraphCompilationError,
    GraphCompiler,
    ResourceAnnotation,
)
from .subgraphs import SubgraphExpansionError, expand_subgraphs
from .topology import CycleError, topological_layers, topological_sort

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
from .validation import GraphDiagnostic, validate_graph

__all__ = [
    "CompileContext",
    "ControlAnnotation",
    "CycleError",
    "DEFAULT_TYPE_REGISTRY",
    "Edge",
    "ExecutionGroup",
    "ExecutionPlan",
    "ExecutionStep",
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
    "TensorHandle",
    "TypeRef",
    "TypeRegistry",
    "canonical_json",
    "content_cache_key",
    "expand_subgraphs",
    "node_cache_key",
    "topological_layers",
    "topological_sort",
    "validate_graph",
]
