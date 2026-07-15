from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import UUID

from comfyng.core.contracts import Contract, register_contract
from comfyng.core.ids import (
    validate_node_id,
    validate_port_name,
    validate_semver,
    validate_sha256,
)
from comfyng.core.json_values import FrozenDict, freeze_json_value
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import ResourceRequirements

from .cache import content_cache_key, node_cache_key
from .subgraphs import SubgraphExpansionError, SubgraphKey, expand_subgraphs
from .topology import (
    critical_path_lengths,
    select_critical_path,
    topological_layers,
)
from .types import (
    DEFAULT_TYPE_REGISTRY,
    Edge,
    Graph,
    NodeInstance,
    OutputBinding,
    TypeRegistry,
)
from .validation import GraphDiagnostic, validate_graph


@dataclass(frozen=True, slots=True)
class CompileContext:
    catalogue: NodeCatalogue = field(default_factory=NodeCatalogue.discover)
    type_registry: TypeRegistry = DEFAULT_TYPE_REGISTRY
    subgraphs: Mapping[SubgraphKey | str, Graph] = field(default_factory=dict)
    max_loop_iterations: int = 1024
    max_subgraph_depth: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.catalogue, NodeCatalogue):
            raise TypeError("catalogue must be a NodeCatalogue")
        if not isinstance(self.type_registry, TypeRegistry):
            raise TypeError("type_registry must be a TypeRegistry")
        if not isinstance(self.subgraphs, Mapping):
            raise TypeError("subgraphs must be a mapping")
        object.__setattr__(self, "subgraphs", MappingProxyType(dict(self.subgraphs)))
        for field_name in ("max_loop_iterations", "max_subgraph_depth"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@register_contract
class ResourceAnnotation(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.graph-resource-annotation"

    gpu: str
    estimated_ram_mb: int
    estimated_vram_mb: int
    network: bool

    def __post_init__(self) -> None:
        if self.gpu not in ("none", "optional", "required"):
            raise ValueError("gpu must be none, optional or required")
        for field_name in ("estimated_ram_mb", "estimated_vram_mb"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if type(self.network) is not bool:
            raise ValueError("network must be a boolean")


@register_contract
class ControlAnnotation(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.graph-control-annotation"

    kind: str
    max_iterations: int | None = None
    selected_branch: str | None = None
    branches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("loop", "condition"):
            raise ValueError("control kind must be loop or condition")
        if self.max_iterations is not None and (
            type(self.max_iterations) is not int or self.max_iterations < 0
        ):
            raise ValueError("max_iterations must be non-negative")
        if (
            self.selected_branch is not None
            and self.selected_branch not in self.branches
        ):
            raise ValueError("selected_branch must be one of branches")


@register_contract
class ExecutionStep(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.execution-step"

    index: int
    group_index: int
    node_id: UUID
    node_type_id: str
    node_type_version: str
    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any]
    incoming_edges: tuple[Edge, ...]
    dependencies: tuple[UUID, ...]
    dependents: tuple[UUID, ...]
    cache_key: str
    is_constant: bool
    cacheable: bool
    critical_path_length: int
    on_critical_path: bool
    resources: ResourceAnnotation
    control: ControlAnnotation | None = None

    def __post_init__(self) -> None:
        for field_name in ("index", "group_index"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not isinstance(self.node_id, UUID):
            raise ValueError("node_id must be a UUID")
        validate_node_id(self.node_type_id)
        validate_semver(self.node_type_version, field="node_type_version")
        for field_name in ("inputs", "metadata"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} must be a JSON object")
            object.__setattr__(
                self,
                field_name,
                freeze_json_value(value, path=f"$.{field_name}"),
            )
        if any(not isinstance(edge, Edge) for edge in self.incoming_edges):
            raise ValueError("incoming_edges must contain Edge values")
        validate_sha256(self.cache_key)
        if type(self.is_constant) is not bool or type(self.cacheable) is not bool:
            raise ValueError("constant/cacheable annotations must be booleans")
        if type(self.critical_path_length) is not int or self.critical_path_length <= 0:
            raise ValueError("critical_path_length must be positive")
        if type(self.on_critical_path) is not bool:
            raise ValueError("on_critical_path must be a boolean")
        if not isinstance(self.resources, ResourceAnnotation):
            raise ValueError("resources must be a ResourceAnnotation")
        if self.control is not None and not isinstance(self.control, ControlAnnotation):
            raise ValueError("control must be a ControlAnnotation")

    @property
    def type_id(self) -> str:
        return self.node_type_id

    @property
    def type_version(self) -> str:
        return self.node_type_version


@register_contract
class ExecutionGroup(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.execution-group"

    index: int
    steps: tuple[ExecutionStep, ...]
    dependency_groups: tuple[int, ...]
    resources: ResourceAnnotation
    critical_path_length: int

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be non-negative")
        if any(not isinstance(step, ExecutionStep) for step in self.steps):
            raise ValueError("steps must contain ExecutionStep values")
        if any(type(value) is not int or value < 0 for value in self.dependency_groups):
            raise ValueError("dependency_groups must be non-negative integers")
        if not isinstance(self.resources, ResourceAnnotation):
            raise ValueError("resources must be a ResourceAnnotation")
        if type(self.critical_path_length) is not int or self.critical_path_length <= 0:
            raise ValueError("critical_path_length must be positive")

    @property
    def parallel(self) -> bool:
        return len(self.steps) > 1


@register_contract
class ExecutionPlan(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.execution-plan"

    graph_id: UUID
    graph_version: int
    cache_key: str
    expanded_graph: Graph
    steps: tuple[ExecutionStep, ...]
    groups: tuple[ExecutionGroup, ...]
    topological_order: tuple[UUID, ...]
    critical_path: tuple[UUID, ...]
    peak_resources: ResourceAnnotation
    diagnostics: tuple[GraphDiagnostic, ...]
    outputs: Mapping[str, OutputBinding]

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, UUID):
            raise ValueError("graph_id must be a UUID")
        if type(self.graph_version) is not int or self.graph_version <= 0:
            raise ValueError("graph_version must be positive")
        validate_sha256(self.cache_key)
        if not isinstance(self.expanded_graph, Graph):
            raise ValueError("expanded_graph must be a Graph")
        if any(not isinstance(step, ExecutionStep) for step in self.steps):
            raise ValueError("steps must contain ExecutionStep values")
        if any(not isinstance(group, ExecutionGroup) for group in self.groups):
            raise ValueError("groups must contain ExecutionGroup values")
        if not isinstance(self.peak_resources, ResourceAnnotation):
            raise ValueError("peak_resources must be a ResourceAnnotation")
        if any(not isinstance(item, GraphDiagnostic) for item in self.diagnostics):
            raise ValueError("diagnostics must contain GraphDiagnostic values")
        frozen_outputs: dict[str, OutputBinding] = {}
        for name, binding in self.outputs.items():
            validate_port_name(name, field="plan output name")
            if not isinstance(binding, OutputBinding):
                raise ValueError("outputs must contain OutputBinding values")
            frozen_outputs[name] = binding
        object.__setattr__(self, "outputs", FrozenDict(frozen_outputs))

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)


class GraphCompilationError(ValueError):
    def __init__(self, diagnostics: tuple[GraphDiagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        errors = tuple(item for item in diagnostics if item.severity == "error")
        summary = "; ".join(f"{item.code}: {item.message}" for item in errors)
        super().__init__(summary or "graph compilation failed")


_ZERO_RESOURCES = ResourceAnnotation(
    gpu="none",
    estimated_ram_mb=0,
    estimated_vram_mb=0,
    network=False,
)


def _resource_annotation(value: ResourceRequirements | None) -> ResourceAnnotation:
    if value is None:
        return _ZERO_RESOURCES
    return ResourceAnnotation(
        gpu=value.gpu.value,
        estimated_ram_mb=value.estimated_ram_mb,
        estimated_vram_mb=value.estimated_vram_mb,
        network=value.network,
    )


_GPU_RANK = {"none": 0, "optional": 1, "required": 2}


def _combine_resources(
    resources: tuple[ResourceAnnotation, ...],
    *,
    peak: bool = False,
) -> ResourceAnnotation:
    if not resources:
        return _ZERO_RESOURCES
    return ResourceAnnotation(
        gpu=max((item.gpu for item in resources), key=_GPU_RANK.__getitem__),
        estimated_ram_mb=(
            max(item.estimated_ram_mb for item in resources)
            if peak
            else sum(item.estimated_ram_mb for item in resources)
        ),
        estimated_vram_mb=(
            max(item.estimated_vram_mb for item in resources)
            if peak
            else sum(item.estimated_vram_mb for item in resources)
        ),
        network=any(item.network for item in resources),
    )


def _resource_index(
    catalogue: NodeCatalogue,
) -> dict[tuple[str, str], ResourceRequirements]:
    index: dict[tuple[str, str], ResourceRequirements] = {}
    for manifest in catalogue.manifests:
        for node in manifest.nodes:
            index[(node.id, node.version)] = manifest.resources
    return index


def _control_annotation(node: NodeInstance) -> ControlAnnotation | None:
    if node.type_id == "ng.control.for_each":
        if "items" in node.inputs and isinstance(node.inputs["items"], tuple):
            bound = len(node.inputs["items"])
        else:
            candidate = node.metadata.get("max_iterations")
            bound = candidate if type(candidate) is int else None
        return ControlAnnotation(kind="loop", max_iterations=bound)
    if node.type_id == "ng.control.switch":
        condition = node.inputs.get("condition")
        selected = None
        if type(condition) is bool:
            selected = "true_value" if condition else "false_value"
        return ControlAnnotation(
            kind="condition",
            selected_branch=selected,
            branches=("true_value", "false_value"),
        )
    return None


def _coerce_context(context: CompileContext | NodeCatalogue | None) -> CompileContext:
    if context is None:
        return CompileContext()
    if isinstance(context, CompileContext):
        return context
    if isinstance(context, NodeCatalogue):
        return CompileContext(catalogue=context)
    raise TypeError("context must be CompileContext, NodeCatalogue or None")


class GraphCompiler:
    @staticmethod
    def compile(
        graph: Graph,
        context: CompileContext | NodeCatalogue | None = None,
    ) -> ExecutionPlan:
        resolved_context = _coerce_context(context)
        try:
            expanded = expand_subgraphs(
                graph,
                resolved_context.subgraphs,
                max_depth=resolved_context.max_subgraph_depth,
            )
        except SubgraphExpansionError as exc:
            diagnostics = (
                GraphDiagnostic(
                    code=exc.code,
                    severity="error",
                    message=str(exc),
                    node_id=exc.node_id,
                ),
            )
            raise GraphCompilationError(diagnostics) from exc

        diagnostics = validate_graph(expanded, resolved_context)
        if any(item.severity == "error" for item in diagnostics):
            raise GraphCompilationError(diagnostics)

        layers = topological_layers(expanded)
        order = tuple(node_id for layer in layers for node_id in layer)
        nodes = {node.id: node for node in expanded.nodes}
        incoming: dict[UUID, list[Edge]] = defaultdict(list)
        outgoing: dict[UUID, list[Edge]] = defaultdict(list)
        for edge in expanded.edges:
            incoming[edge.target_node_id].append(edge)
            outgoing[edge.source_node_id].append(edge)
        for values in (*incoming.values(), *outgoing.values()):
            values.sort(
                key=lambda edge: (
                    edge.target_port,
                    str(edge.source_node_id),
                    edge.source_port,
                    str(edge.target_node_id),
                )
            )

        group_indexes = {
            node_id: group_index
            for group_index, layer in enumerate(layers)
            for node_id in layer
        }
        path_lengths = critical_path_lengths(order, expanded.edges)
        critical_path = select_critical_path(order, expanded.edges, path_lengths)
        critical_nodes = set(critical_path)
        external_targets = {
            (binding.node_id, binding.port): name
            for name, binding in expanded.inputs.items()
        }
        constants: dict[UUID, bool] = {}
        cache_keys: dict[UUID, str] = {}
        resource_values = _resource_index(resolved_context.catalogue)
        steps: list[ExecutionStep] = []

        for index, node_id in enumerate(order):
            node = nodes[node_id]
            node_incoming = tuple(incoming.get(node_id, ()))
            dependencies = tuple(
                sorted({edge.source_node_id for edge in node_incoming}, key=str)
            )
            dependents = tuple(
                sorted(
                    {edge.target_node_id for edge in outgoing.get(node_id, ())},
                    key=str,
                )
            )
            has_external_input = any(
                binding_node_id == node_id
                for binding_node_id, _port in external_targets
            )
            is_constant = not has_external_input and all(
                constants[dependency] for dependency in dependencies
            )
            constants[node_id] = is_constant

            upstream: dict[str, object] = {
                edge.target_port: {
                    "cache_key": cache_keys[edge.source_node_id],
                    "source_port": edge.source_port,
                }
                for edge in node_incoming
            }
            for (binding_node_id, port), name in sorted(
                external_targets.items(),
                key=lambda item: (str(item[0][0]), item[0][1]),
            ):
                if binding_node_id == node_id:
                    upstream[port] = {"graph_input": name}
            cache_key = node_cache_key(node, upstream)
            cache_keys[node_id] = cache_key
            resources = _resource_annotation(
                resource_values.get((node.type_id, node.type_version))
            )
            steps.append(
                ExecutionStep(
                    index=index,
                    group_index=group_indexes[node_id],
                    node_id=node_id,
                    node_type_id=node.type_id,
                    node_type_version=node.type_version,
                    inputs=node.inputs,
                    metadata=node.metadata,
                    incoming_edges=node_incoming,
                    dependencies=dependencies,
                    dependents=dependents,
                    cache_key=cache_key,
                    is_constant=is_constant,
                    cacheable=is_constant and not resources.network,
                    critical_path_length=path_lengths[node_id],
                    on_critical_path=node_id in critical_nodes,
                    resources=resources,
                    control=_control_annotation(node),
                )
            )

        steps_by_id = {step.node_id: step for step in steps}
        groups: list[ExecutionGroup] = []
        for group_index, layer in enumerate(layers):
            group_steps = tuple(steps_by_id[node_id] for node_id in layer)
            dependency_groups = tuple(
                sorted(
                    {
                        group_indexes[dependency]
                        for step in group_steps
                        for dependency in step.dependencies
                        if group_indexes[dependency] != group_index
                    }
                )
            )
            groups.append(
                ExecutionGroup(
                    index=group_index,
                    steps=group_steps,
                    dependency_groups=dependency_groups,
                    resources=_combine_resources(
                        tuple(step.resources for step in group_steps)
                    ),
                    critical_path_length=max(
                        step.critical_path_length for step in group_steps
                    ),
                )
            )
        peak_resources = _combine_resources(
            tuple(group.resources for group in groups),
            peak=True,
        )
        output_content = {
            name: {
                "cache_key": cache_keys[binding.node_id],
                "port": binding.port,
            }
            for name, binding in sorted(expanded.outputs.items())
        }
        plan_key = content_cache_key(
            {
                "plan_version": 1,
                "graph_version": expanded.version,
                "steps": sorted(step.cache_key for step in steps),
                "outputs": output_content,
            }
        )
        return ExecutionPlan(
            graph_id=expanded.id,
            graph_version=expanded.version,
            cache_key=plan_key,
            expanded_graph=expanded,
            steps=tuple(steps),
            groups=tuple(groups),
            topological_order=order,
            critical_path=critical_path,
            peak_resources=peak_resources,
            diagnostics=diagnostics,
            outputs=expanded.outputs,
        )
