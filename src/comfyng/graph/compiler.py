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
    validate_package_id,
    validate_port_name,
    validate_semver,
    validate_sha256,
)
from comfyng.core.json_values import FrozenDict, freeze_json_value
from comfyng.plugins.catalogue import NodeCatalogue
from comfyng.plugins.manifest import (
    NodeDefinition,
    PluginManifest,
    ResourceRequirements,
)

from .cache import content_cache_key, node_cache_key
from .subgraphs import (
    SubgraphExpansionError,
    SubgraphKey,
    SubgraphTrace,
    expand_subgraphs_with_trace,
)
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
from .validation import GraphDiagnostic, validate_graph, validate_graph_structure


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


def _validate_step_id_tuple(
    values: tuple[UUID, ...],
    *,
    field_name: str,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, UUID) for value in values):
        raise ValueError(f"{field_name} must contain UUID values")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")


@register_contract
class ControlBranch(Contract):
    """One executable branch owned by a condition controller."""

    TYPE_ID: ClassVar[str] = "comfyng.control-branch"

    name: str
    source_port: str
    member_step_ids: tuple[UUID, ...]
    entry_step_ids: tuple[UUID, ...]
    exit_step_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        validate_port_name(self.name, field="branch name")
        validate_port_name(self.source_port, field="branch source port")
        for field_name in (
            "member_step_ids",
            "entry_step_ids",
            "exit_step_ids",
        ):
            _validate_step_id_tuple(getattr(self, field_name), field_name=field_name)
        members = set(self.member_step_ids)
        if not set(self.entry_step_ids).issubset(members):
            raise ValueError("branch entry_step_ids must be branch members")
        if not set(self.exit_step_ids).issubset(members):
            raise ValueError("branch exit_step_ids must be branch members")
        if members and (not self.entry_step_ids or not self.exit_step_ids):
            raise ValueError("non-empty branches require entry and exit steps")


@register_contract
class ControlRegion(Contract):
    """Executable condition or bounded-loop region in an execution plan."""

    TYPE_ID: ClassVar[str] = "comfyng.control-region"

    kind: str
    controller_step_id: UUID
    parent_controller_step_id: UUID | None
    nesting_depth: int
    branches: tuple[ControlBranch, ...] = ()
    iterable_step_ids: tuple[UUID, ...] = ()
    body_step_ids: tuple[UUID, ...] = ()
    boundary_step_ids: tuple[UUID, ...] = ()
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("condition", "loop"):
            raise ValueError("control region kind must be condition or loop")
        if not isinstance(self.controller_step_id, UUID):
            raise ValueError("controller_step_id must be a UUID")
        if self.parent_controller_step_id is not None and not isinstance(
            self.parent_controller_step_id, UUID
        ):
            raise ValueError("parent_controller_step_id must be a UUID")
        if self.parent_controller_step_id == self.controller_step_id:
            raise ValueError("a control region cannot be its own parent")
        if type(self.nesting_depth) is not int or self.nesting_depth < 0:
            raise ValueError("nesting_depth must be non-negative")
        if (self.parent_controller_step_id is None) != (self.nesting_depth == 0):
            raise ValueError("only root control regions may have depth zero")
        if any(not isinstance(branch, ControlBranch) for branch in self.branches):
            raise ValueError("branches must contain ControlBranch values")
        for field_name in (
            "iterable_step_ids",
            "body_step_ids",
            "boundary_step_ids",
        ):
            _validate_step_id_tuple(getattr(self, field_name), field_name=field_name)
        owned_sets = (
            set(self.iterable_step_ids),
            set(self.body_step_ids),
            set(self.boundary_step_ids),
        )
        if any(self.controller_step_id in values for values in owned_sets):
            raise ValueError("controller step cannot be a region member")
        if self.kind == "condition":
            if len(self.branches) < 2:
                raise ValueError("condition regions require at least two branches")
            names = tuple(branch.name for branch in self.branches)
            if len(set(names)) != len(names):
                raise ValueError("condition branch names must be unique")
            branch_members: set[UUID] = set()
            for branch in self.branches:
                overlap = branch_members.intersection(branch.member_step_ids)
                if overlap:
                    raise ValueError("condition branch members must be disjoint")
                branch_members.update(branch.member_step_ids)
            if any(owned_sets) or self.max_iterations is not None:
                raise ValueError("condition regions cannot define loop fields")
        else:
            if self.branches:
                raise ValueError("loop regions cannot define branches")
            if type(self.max_iterations) is not int or self.max_iterations < 0:
                raise ValueError("loop regions require a non-negative max_iterations")
            if set(self.iterable_step_ids).intersection(self.body_step_ids):
                raise ValueError("loop iterable and body steps must be disjoint")
            if set(self.body_step_ids).intersection(self.boundary_step_ids):
                raise ValueError("loop body and boundary steps must be disjoint")


@register_contract
class FusionEvidence(Contract):
    """Auditable proof that a sequence of nodes may execute as one step."""

    TYPE_ID: ClassVar[str] = "comfyng.graph-fusion-evidence"

    kind: str
    member_node_ids: tuple[UUID, ...]
    links: tuple[Edge, ...]
    package_id: str
    package_version: str
    runtime_entrypoint: str
    runtime_isolation: str
    resource_fingerprint: str
    schema_fingerprints: tuple[str, ...]
    pure: bool
    deterministic: bool
    one_to_one: bool
    control_free: bool

    def __post_init__(self) -> None:
        validate_port_name(self.kind, field="fusion kind")
        if len(self.member_node_ids) < 2:
            raise ValueError("fusion must contain at least two nodes")
        if any(not isinstance(node_id, UUID) for node_id in self.member_node_ids):
            raise ValueError("fusion member_node_ids must contain UUID values")
        if len(set(self.member_node_ids)) != len(self.member_node_ids):
            raise ValueError("fusion member_node_ids must be unique")
        if len(self.links) != len(self.member_node_ids) - 1:
            raise ValueError("fusion links must connect every adjacent member")
        for index, link in enumerate(self.links):
            if not isinstance(link, Edge):
                raise ValueError("fusion links must contain Edge values")
            if (
                link.source_node_id != self.member_node_ids[index]
                or link.target_node_id != self.member_node_ids[index + 1]
            ):
                raise ValueError("fusion links must follow member_node_ids")
        validate_package_id(self.package_id)
        validate_semver(self.package_version, field="fusion package_version")
        if not self.runtime_entrypoint or ":" not in self.runtime_entrypoint:
            raise ValueError("fusion runtime_entrypoint must be an entrypoint")
        if self.runtime_isolation not in (
            "plugin_worker",
            "gpu_model_worker",
            "cpu_worker",
            "io_worker",
            "isolated_process",
        ):
            raise ValueError("fusion runtime_isolation is invalid")
        validate_sha256(self.resource_fingerprint)
        if len(self.schema_fingerprints) != len(self.links):
            raise ValueError("fusion requires one schema fingerprint per link")
        for fingerprint in self.schema_fingerprints:
            validate_sha256(fingerprint)
        if not (
            type(self.pure) is bool
            and type(self.deterministic) is bool
            and type(self.one_to_one) is bool
            and type(self.control_free) is bool
        ):
            raise ValueError("fusion safety evidence must be boolean")
        if not all((self.pure, self.deterministic, self.one_to_one, self.control_free)):
            raise ValueError("fusion safety evidence must be affirmative")


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
    member_node_ids: tuple[UUID, ...] = ()
    fusion: FusionEvidence | None = None

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
            setattr(
                self,
                field_name,
                freeze_json_value(value, path=f"$.{field_name}"),
            )
        if any(not isinstance(edge, Edge) for edge in self.incoming_edges):
            raise ValueError("incoming_edges must contain Edge values")
        if len(set(self.incoming_edges)) != len(self.incoming_edges):
            raise ValueError("incoming_edges must be unique")
        for field_name in ("dependencies", "dependents"):
            values = getattr(self, field_name)
            _validate_step_id_tuple(values, field_name=field_name)
            if self.node_id in values:
                raise ValueError(f"{field_name} cannot contain the step itself")
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
        members = self.member_node_ids or (self.node_id,)
        if any(not isinstance(member, UUID) for member in members):
            raise ValueError("member_node_ids must contain UUID values")
        if len(set(members)) != len(members):
            raise ValueError("member_node_ids must be unique")
        if members[0] != self.node_id:
            raise ValueError("node_id must be the first member_node_id")
        self.member_node_ids = members
        member_set = set(members)
        if member_set.intersection(self.dependencies):
            raise ValueError("dependencies cannot contain fused member nodes")
        if member_set.intersection(self.dependents):
            raise ValueError("dependents cannot contain fused member nodes")
        if any(edge.target_node_id not in member_set for edge in self.incoming_edges):
            raise ValueError("incoming_edges must target a step member")
        if any(edge.source_node_id in member_set for edge in self.incoming_edges):
            raise ValueError("incoming_edges cannot contain internal fused links")
        if self.fusion is None:
            if len(members) != 1:
                raise ValueError("multi-node steps require fusion evidence")
        elif not isinstance(self.fusion, FusionEvidence):
            raise ValueError("fusion must be FusionEvidence")
        elif self.fusion.member_node_ids != members:
            raise ValueError("fusion members must match step members")
        if self.control is not None and self.fusion is not None:
            raise ValueError("control steps cannot be fused")

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
        if not self.steps:
            raise ValueError("execution groups must contain at least one step")
        if any(step.group_index != self.index for step in self.steps):
            raise ValueError("step group_index must match its execution group")
        member_ids = tuple(
            member for step in self.steps for member in step.member_node_ids
        )
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("execution group steps must have unique members")
        if any(type(value) is not int or value < 0 for value in self.dependency_groups):
            raise ValueError("dependency_groups must be non-negative integers")
        if len(set(self.dependency_groups)) != len(self.dependency_groups):
            raise ValueError("dependency_groups must be unique")
        if tuple(sorted(self.dependency_groups)) != self.dependency_groups:
            raise ValueError("dependency_groups must be sorted")
        if any(value >= self.index for value in self.dependency_groups):
            raise ValueError("dependency_groups must precede the current group")
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
    fusions: tuple[FusionEvidence, ...] = ()
    control_regions: tuple[ControlRegion, ...] = ()
    subgraph_traces: tuple[SubgraphTrace, ...] = ()

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
        if any(not isinstance(item, FusionEvidence) for item in self.fusions):
            raise ValueError("fusions must contain FusionEvidence values")
        if any(not isinstance(item, ControlRegion) for item in self.control_regions):
            raise ValueError("control_regions must contain ControlRegion values")
        if any(not isinstance(item, SubgraphTrace) for item in self.subgraph_traces):
            raise ValueError("subgraph_traces must contain SubgraphTrace values")

        expanded_ids = tuple(node.id for node in self.expanded_graph.nodes)
        expanded_id_set = set(expanded_ids)
        if self.graph_id != self.expanded_graph.id:
            raise ValueError("graph_id must match expanded_graph")
        if self.graph_version != self.expanded_graph.version:
            raise ValueError("graph_version must match expanded_graph")
        if (
            len(set(self.topological_order)) != len(self.topological_order)
            or set(self.topological_order) != expanded_id_set
        ):
            raise ValueError(
                "topological_order must contain every expanded graph node exactly once"
            )
        topological_indexes = {
            node_id: index for index, node_id in enumerate(self.topological_order)
        }
        if any(
            topological_indexes[edge.source_node_id]
            >= topological_indexes[edge.target_node_id]
            for edge in self.expanded_graph.edges
        ):
            raise ValueError("topological_order does not respect graph edges")
        if any(not isinstance(node_id, UUID) for node_id in self.critical_path):
            raise ValueError("critical_path must contain UUID values")
        if len(set(self.critical_path)) != len(self.critical_path):
            raise ValueError("critical_path must contain unique nodes")
        if not set(self.critical_path).issubset(expanded_id_set):
            raise ValueError("critical_path must reference expanded graph nodes")
        if (
            tuple(sorted(self.critical_path, key=topological_indexes.__getitem__))
            != self.critical_path
        ):
            raise ValueError("critical_path must follow topological_order")

        if tuple(step.index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("execution step indexes must be contiguous")
        representative_ids = tuple(step.node_id for step in self.steps)
        if len(set(representative_ids)) != len(representative_ids):
            raise ValueError("execution step node ids must be unique")
        all_members = tuple(
            member for step in self.steps for member in step.member_node_ids
        )
        if (
            len(set(all_members)) != len(all_members)
            or set(all_members) != expanded_id_set
        ):
            raise ValueError("execution steps must partition expanded graph nodes")
        step_by_id = {step.node_id: step for step in self.steps}
        for step in self.steps:
            if not set(step.dependencies).issubset(step_by_id):
                raise ValueError("step dependencies must reference execution steps")
            if not set(step.dependents).issubset(step_by_id):
                raise ValueError("step dependents must reference execution steps")
            for dependency in step.dependencies:
                if step.node_id not in step_by_id[dependency].dependents:
                    raise ValueError("step dependencies/dependents must be symmetric")
            for dependent in step.dependents:
                if step.node_id not in step_by_id[dependent].dependencies:
                    raise ValueError("step dependencies/dependents must be symmetric")

        if tuple(group.index for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ValueError("execution groups must have contiguous indexes")
        grouped_steps = tuple(step for group in self.groups for step in group.steps)
        if grouped_steps != self.steps:
            raise ValueError("execution groups must partition plan steps in order")
        group_for_step = {
            step.node_id: group.index for group in self.groups for step in group.steps
        }
        for group in self.groups:
            expected_dependencies = tuple(
                sorted(
                    {
                        group_for_step[dependency]
                        for step in group.steps
                        for dependency in step.dependencies
                        if group_for_step[dependency] != group.index
                    }
                )
            )
            if group.dependency_groups != expected_dependencies:
                raise ValueError("execution groups have inconsistent dependencies")
        step_fusions = tuple(
            step.fusion for step in self.steps if step.fusion is not None
        )
        if self.fusions != step_fusions:
            raise ValueError("plan fusions must match fused steps")
        control_steps = {
            step.node_id: step for step in self.steps if step.control is not None
        }
        region_controllers = tuple(
            region.controller_step_id for region in self.control_regions
        )
        if len(set(region_controllers)) != len(region_controllers):
            raise ValueError("control region controllers must be unique")
        if set(region_controllers) != set(control_steps):
            raise ValueError("control regions must match control execution steps")
        regions_by_controller = {
            region.controller_step_id: region for region in self.control_regions
        }
        for region in self.control_regions:
            step = control_steps[region.controller_step_id]
            assert step.control is not None
            if step.control.kind != region.kind:
                raise ValueError("control region kind must match its controller")
            references = set(region.iterable_step_ids)
            references.update(region.body_step_ids)
            references.update(region.boundary_step_ids)
            for branch in region.branches:
                references.update(branch.member_step_ids)
            if not references.issubset(step_by_id):
                raise ValueError("control regions must reference execution steps")
            parent_id = region.parent_controller_step_id
            if parent_id is None:
                if region.nesting_depth != 0:
                    raise ValueError("root control region depth must be zero")
            else:
                parent = regions_by_controller.get(parent_id)
                if parent is None:
                    raise ValueError("control region parent must exist")
                if region.nesting_depth != parent.nesting_depth + 1:
                    raise ValueError("control region nesting depth is inconsistent")

        trace_by_call = {trace.call_node_id: trace for trace in self.subgraph_traces}
        if len(trace_by_call) != len(self.subgraph_traces):
            raise ValueError("subgraph trace call ids must be unique")
        for trace in self.subgraph_traces:
            if not set(trace.member_node_ids).issubset(expanded_id_set):
                raise ValueError("subgraph traces must reference expanded graph nodes")
            if trace.parent_call_node_id is None:
                if trace.depth != 0:
                    raise ValueError("root subgraph trace depth must be zero")
            else:
                parent = trace_by_call.get(trace.parent_call_node_id)
                if parent is None:
                    raise ValueError("subgraph trace parent must exist")
                if trace.depth != parent.depth + 1:
                    raise ValueError("subgraph trace depth is inconsistent")
                if not set(trace.member_node_ids).issubset(parent.member_node_ids):
                    raise ValueError(
                        "nested subgraph members must belong to their parent"
                    )
        frozen_outputs: dict[str, OutputBinding] = {}
        for name, binding in self.outputs.items():
            validate_port_name(name, field="plan output name")
            if not isinstance(binding, OutputBinding):
                raise ValueError("outputs must contain OutputBinding values")
            if binding.node_id not in expanded_id_set:
                raise ValueError("plan outputs must reference expanded graph nodes")
            frozen_outputs[name] = binding
        if frozen_outputs != dict(self.expanded_graph.outputs):
            raise ValueError("plan outputs must match expanded_graph outputs")
        self.outputs = FrozenDict(frozen_outputs)

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def step_for_node(self, node_id: UUID) -> ExecutionStep:
        if not isinstance(node_id, UUID):
            raise TypeError("node_id must be a UUID")
        for step in self.steps:
            if node_id in step.member_node_ids:
                return step
        raise KeyError(node_id)


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


@dataclass(frozen=True, slots=True)
class _FusionProfile:
    definition: NodeDefinition
    manifest: PluginManifest
    resource_fingerprint: str
    runtime_fingerprint: str


@dataclass(frozen=True, slots=True)
class _FusionUnit:
    member_node_ids: tuple[UUID, ...]
    evidence: FusionEvidence | None = None

    @property
    def node_id(self) -> UUID:
        return self.member_node_ids[0]


def _fusion_profile_index(
    catalogue: NodeCatalogue,
) -> dict[tuple[str, str], _FusionProfile]:
    profiles: dict[tuple[str, str], _FusionProfile] = {}
    for manifest in catalogue.manifests:
        resource_fingerprint = content_cache_key(manifest.resources)
        runtime_fingerprint = content_cache_key(
            {
                "package_id": manifest.package.id,
                "package_version": manifest.package.version,
                "runtime": manifest.runtime,
                "resources": manifest.resources,
            }
        )
        for manifest_definition in manifest.nodes:
            key = (manifest_definition.id, manifest_definition.version)
            try:
                definition = catalogue.get(*key)
            except ValueError:
                continue
            if (
                definition.package_id != manifest.package.id
                or manifest_definition.package_id != manifest.package.id
            ):
                continue
            profiles[key] = _FusionProfile(
                definition=definition,
                manifest=manifest,
                resource_fingerprint=resource_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
            )
    return profiles


def _port_schema(
    definition: NodeDefinition,
    *,
    output: bool,
    port: str,
) -> object | None:
    schema = definition.output_schema if output else definition.input_schema
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    return properties.get(port)


def _fusion_link_evidence(
    edge: Edge,
    *,
    nodes: Mapping[UUID, NodeInstance],
    incoming: Mapping[UUID, list[Edge]],
    outgoing: Mapping[UUID, list[Edge]],
    steps: Mapping[UUID, ExecutionStep],
    profiles: Mapping[tuple[str, str], _FusionProfile],
    graph_input_nodes: frozenset[UUID],
    graph_output_nodes: frozenset[UUID],
) -> tuple[_FusionProfile, str, str] | None:
    source = nodes[edge.source_node_id]
    target = nodes[edge.target_node_id]
    source_definition = profiles.get((source.type_id, source.type_version))
    target_definition = profiles.get((target.type_id, target.type_version))
    if source_definition is None or target_definition is None:
        return None
    source_traits = source_definition.definition.execution
    target_traits = target_definition.definition.execution
    kind = source_traits.fusion_kind
    if kind is None or kind != target_traits.fusion_kind:
        return None
    if not all(
        (
            source_traits.pure,
            source_traits.deterministic,
            target_traits.pure,
            target_traits.deterministic,
        )
    ):
        return None
    if source_traits.side_effects or target_traits.side_effects:
        return None
    if len(outgoing.get(source.id, ())) != 1:
        return None
    if len(incoming.get(target.id, ())) != 1:
        return None
    if source.id in graph_output_nodes or target.id in graph_input_nodes:
        return None
    if steps[source.id].control is not None or steps[target.id].control is not None:
        return None
    if source_definition.runtime_fingerprint != target_definition.runtime_fingerprint:
        return None
    if source_definition.manifest.resources.network:
        return None
    source_schema = _port_schema(
        source_definition.definition,
        output=True,
        port=edge.source_port,
    )
    target_schema = _port_schema(
        target_definition.definition,
        output=False,
        port=edge.target_port,
    )
    if source_schema is None or target_schema is None:
        return None
    source_schema_fingerprint = content_cache_key(source_schema)
    if source_schema_fingerprint != content_cache_key(target_schema):
        return None
    return source_definition, kind, source_schema_fingerprint


def _fusion_units(
    graph: Graph,
    order: tuple[UUID, ...],
    *,
    nodes: Mapping[UUID, NodeInstance],
    incoming: Mapping[UUID, list[Edge]],
    outgoing: Mapping[UUID, list[Edge]],
    steps: Mapping[UUID, ExecutionStep],
    catalogue: NodeCatalogue,
) -> tuple[_FusionUnit, ...]:
    profiles = _fusion_profile_index(catalogue)
    graph_input_nodes = frozenset(binding.node_id for binding in graph.inputs.values())
    graph_output_nodes = frozenset(
        binding.node_id for binding in graph.outputs.values()
    )
    consumed: set[UUID] = set()
    units: list[_FusionUnit] = []
    for node_id in order:
        if node_id in consumed:
            continue
        members = [node_id]
        links: list[Edge] = []
        schema_fingerprints: list[str] = []
        profile: _FusionProfile | None = None
        kind: str | None = None
        current_id = node_id
        while len(outgoing.get(current_id, ())) == 1:
            edge = outgoing[current_id][0]
            if edge.target_node_id in consumed:
                break
            evidence = _fusion_link_evidence(
                edge,
                nodes=nodes,
                incoming=incoming,
                outgoing=outgoing,
                steps=steps,
                profiles=profiles,
                graph_input_nodes=graph_input_nodes,
                graph_output_nodes=graph_output_nodes,
            )
            if evidence is None:
                break
            candidate_profile, candidate_kind, schema_fingerprint = evidence
            if profile is not None and (
                candidate_profile.runtime_fingerprint != profile.runtime_fingerprint
                or candidate_kind != kind
            ):
                break
            profile = candidate_profile
            kind = candidate_kind
            links.append(edge)
            schema_fingerprints.append(schema_fingerprint)
            current_id = edge.target_node_id
            members.append(current_id)

        if len(members) == 1:
            units.append(_FusionUnit(member_node_ids=(node_id,)))
            consumed.add(node_id)
            continue
        assert profile is not None and kind is not None
        member_node_ids = tuple(members)
        evidence = FusionEvidence(
            kind=kind,
            member_node_ids=member_node_ids,
            links=tuple(links),
            package_id=profile.manifest.package.id,
            package_version=profile.manifest.package.version,
            runtime_entrypoint=profile.manifest.runtime.entrypoint,
            runtime_isolation=profile.manifest.runtime.isolation.value,
            resource_fingerprint=profile.resource_fingerprint,
            schema_fingerprints=tuple(schema_fingerprints),
            pure=True,
            deterministic=True,
            one_to_one=True,
            control_free=True,
        )
        units.append(_FusionUnit(member_node_ids=member_node_ids, evidence=evidence))
        consumed.update(member_node_ids)
    return tuple(units)


def _collapse_fused_steps(
    graph: Graph,
    order: tuple[UUID, ...],
    base_steps: tuple[ExecutionStep, ...],
    catalogue: NodeCatalogue,
) -> tuple[tuple[ExecutionStep, ...], tuple[tuple[UUID, ...], ...]]:
    nodes = {node.id: node for node in graph.nodes}
    incoming: dict[UUID, list[Edge]] = defaultdict(list)
    outgoing: dict[UUID, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target_node_id].append(edge)
        outgoing[edge.source_node_id].append(edge)
    base_by_id = {step.node_id: step for step in base_steps}
    units = _fusion_units(
        graph,
        order,
        nodes=nodes,
        incoming=incoming,
        outgoing=outgoing,
        steps=base_by_id,
        catalogue=catalogue,
    )
    representative_for = {
        member: unit.node_id for unit in units for member in unit.member_node_ids
    }
    synthetic_edges = tuple(
        Edge(
            source_node_id=representative_for[edge.source_node_id],
            source_port=edge.source_port,
            target_node_id=representative_for[edge.target_node_id],
            target_port=edge.target_port,
        )
        for edge in graph.edges
        if representative_for[edge.source_node_id]
        != representative_for[edge.target_node_id]
    )
    unit_layers = topological_layers(
        tuple(unit.node_id for unit in units),
        synthetic_edges,
    )
    unit_group_indexes = {
        node_id: group_index
        for group_index, layer in enumerate(unit_layers)
        for node_id in layer
    }
    unit_order = tuple(node_id for layer in unit_layers for node_id in layer)
    order_indexes = {node_id: index for index, node_id in enumerate(unit_order)}
    units_by_id = {unit.node_id: unit for unit in units}
    external_incoming: dict[UUID, list[Edge]] = defaultdict(list)
    external_outgoing: dict[UUID, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        source_unit = representative_for[edge.source_node_id]
        target_unit = representative_for[edge.target_node_id]
        if source_unit != target_unit:
            external_incoming[target_unit].append(edge)
            external_outgoing[source_unit].append(edge)
    collapsed: list[ExecutionStep] = []
    for node_id in unit_order:
        unit = units_by_id[node_id]
        members = tuple(base_by_id[member] for member in unit.member_node_ids)
        dependencies = tuple(
            sorted(
                {
                    representative_for[edge.source_node_id]
                    for edge in external_incoming.get(node_id, ())
                },
                key=str,
            )
        )
        dependents = tuple(
            sorted(
                {
                    representative_for[edge.target_node_id]
                    for edge in external_outgoing.get(node_id, ())
                },
                key=str,
            )
        )
        first = members[0]
        cache_key = first.cache_key
        if unit.evidence is not None:
            cache_key = content_cache_key(
                {
                    "fusion": unit.evidence,
                    "member_cache_keys": tuple(member.cache_key for member in members),
                }
            )
        collapsed.append(
            ExecutionStep(
                index=order_indexes[node_id],
                group_index=unit_group_indexes[node_id],
                node_id=node_id,
                node_type_id=first.node_type_id,
                node_type_version=first.node_type_version,
                inputs=first.inputs,
                metadata=first.metadata,
                incoming_edges=tuple(external_incoming.get(node_id, ())),
                dependencies=dependencies,
                dependents=dependents,
                cache_key=cache_key,
                is_constant=all(member.is_constant for member in members),
                cacheable=all(member.cacheable for member in members),
                critical_path_length=max(
                    member.critical_path_length for member in members
                ),
                on_critical_path=any(member.on_critical_path for member in members),
                resources=_combine_resources(
                    tuple(member.resources for member in members),
                    peak=True,
                ),
                control=first.control if len(members) == 1 else None,
                member_node_ids=unit.member_node_ids,
                fusion=unit.evidence,
            )
        )
    return tuple(collapsed), unit_layers


def _closure(
    start: set[UUID],
    adjacency: Mapping[UUID, set[UUID]],
) -> set[UUID]:
    visited: set[UUID] = set()
    pending = list(sorted(start, key=str, reverse=True))
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(
            node
            for node in sorted(adjacency.get(node_id, ()), key=str, reverse=True)
            if node not in visited
        )
    return visited


def _ordered_representatives(
    node_ids: set[UUID],
    *,
    representative_for: Mapping[UUID, UUID],
    step_indexes: Mapping[UUID, int],
) -> tuple[UUID, ...]:
    return tuple(
        sorted(
            {
                representative_for[node_id]
                for node_id in node_ids
                if node_id in representative_for
            },
            key=step_indexes.__getitem__,
        )
    )


@dataclass(frozen=True, slots=True)
class _ControlRegionDraft:
    kind: str
    controller_step_id: UUID
    branches: tuple[ControlBranch, ...] = ()
    iterable_step_ids: tuple[UUID, ...] = ()
    body_step_ids: tuple[UUID, ...] = ()
    boundary_step_ids: tuple[UUID, ...] = ()
    max_iterations: int | None = None

    @property
    def member_step_ids(self) -> frozenset[UUID]:
        members = set(self.iterable_step_ids)
        members.update(self.body_step_ids)
        members.update(self.boundary_step_ids)
        for branch in self.branches:
            members.update(branch.member_step_ids)
        return frozenset(members)


def _derive_control_regions(
    graph: Graph,
    steps: tuple[ExecutionStep, ...],
) -> tuple[tuple[ControlRegion, ...], tuple[GraphDiagnostic, ...]]:
    nodes = {node.id: node for node in graph.nodes}
    incoming_nodes: dict[UUID, set[UUID]] = defaultdict(set)
    outgoing_nodes: dict[UUID, set[UUID]] = defaultdict(set)
    incoming_edges: dict[UUID, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        incoming_nodes[edge.target_node_id].add(edge.source_node_id)
        outgoing_nodes[edge.source_node_id].add(edge.target_node_id)
        incoming_edges[edge.target_node_id].append(edge)

    representative_for = {
        member: step.node_id for step in steps for member in step.member_node_ids
    }
    steps_by_id = {step.node_id: step for step in steps}
    step_indexes = {step.node_id: step.index for step in steps}
    loop_nodes = tuple(
        node for node in graph.nodes if node.type_id == "ng.control.for_each"
    )
    raw_loop_bodies: dict[UUID, set[UUID]] = {}
    raw_loop_boundaries: dict[UUID, set[UUID]] = {}
    raw_loop_iterables: dict[UUID, set[UUID]] = {}
    diagnostics: list[GraphDiagnostic] = []

    for loop in loop_nodes:
        descendants = _closure(set(outgoing_nodes.get(loop.id, ())), outgoing_nodes)
        boundaries = {
            node_id
            for node_id in descendants
            if nodes[node_id].type_id == "ng.control.collect"
        }
        if descendants and not boundaries:
            diagnostics.append(
                GraphDiagnostic(
                    code="malformed_loop_region",
                    severity="error",
                    message="a for-each body must terminate at an NG Collect boundary",
                    node_id=loop.id,
                )
            )
        body: set[UUID] = set()
        for boundary in boundaries:
            body.update(
                _closure(
                    set(incoming_nodes.get(boundary, ())), incoming_nodes
                ).intersection(descendants)
            )
        body.difference_update(boundaries)
        body.discard(loop.id)
        item_sources = {
            edge.source_node_id
            for edge in incoming_edges.get(loop.id, ())
            if edge.target_port == "items"
        }
        iterable = _closure(item_sources, incoming_nodes)
        iterable.discard(loop.id)
        raw_loop_bodies[loop.id] = body
        raw_loop_boundaries[loop.id] = boundaries
        raw_loop_iterables[loop.id] = iterable

    drafts: list[_ControlRegionDraft] = []
    for step in steps:
        if step.control is None:
            continue
        node = nodes[step.node_id]
        if step.control.kind == "condition":
            branch_names = tuple(sorted(step.control.branches))
            branch_nodes: dict[str, set[UUID]] = {}
            for name in branch_names:
                sources = {
                    edge.source_node_id
                    for edge in incoming_edges.get(node.id, ())
                    if edge.target_port == name
                }
                branch_nodes[name] = _closure(sources, incoming_nodes)
            shared = (
                set.intersection(*(set(values) for values in branch_nodes.values()))
                if branch_nodes
                else set()
            )
            condition_sources = {
                edge.source_node_id
                for edge in incoming_edges.get(node.id, ())
                if edge.target_port == "condition"
            }
            excluded = shared.union(_closure(condition_sources, incoming_nodes))
            excluded.add(node.id)
            for loop_id, body in raw_loop_bodies.items():
                if node.id in body:
                    excluded.add(loop_id)

            branches: list[ControlBranch] = []
            for name in branch_names:
                member_steps = _ordered_representatives(
                    branch_nodes[name].difference(excluded),
                    representative_for=representative_for,
                    step_indexes=step_indexes,
                )
                member_set = set(member_steps)
                entries = tuple(
                    member
                    for member in member_steps
                    if not member_set.intersection(steps_by_id[member].dependencies)
                )
                exits = tuple(
                    member
                    for member in member_steps
                    if not member_set.intersection(steps_by_id[member].dependents)
                )
                branches.append(
                    ControlBranch(
                        name=name,
                        source_port=name,
                        member_step_ids=member_steps,
                        entry_step_ids=entries,
                        exit_step_ids=exits,
                    )
                )
            drafts.append(
                _ControlRegionDraft(
                    kind="condition",
                    controller_step_id=step.node_id,
                    branches=tuple(branches),
                )
            )
            continue

        assert step.control.kind == "loop"
        drafts.append(
            _ControlRegionDraft(
                kind="loop",
                controller_step_id=step.node_id,
                iterable_step_ids=_ordered_representatives(
                    raw_loop_iterables.get(step.node_id, set()),
                    representative_for=representative_for,
                    step_indexes=step_indexes,
                ),
                body_step_ids=_ordered_representatives(
                    raw_loop_bodies.get(step.node_id, set()),
                    representative_for=representative_for,
                    step_indexes=step_indexes,
                ),
                boundary_step_ids=_ordered_representatives(
                    raw_loop_boundaries.get(step.node_id, set()),
                    representative_for=representative_for,
                    step_indexes=step_indexes,
                ),
                max_iterations=step.control.max_iterations,
            )
        )

    drafts_by_controller = {draft.controller_step_id: draft for draft in drafts}
    parent_for: dict[UUID, UUID | None] = {}
    for draft in drafts:
        candidates = tuple(
            candidate
            for candidate in drafts
            if candidate.controller_step_id != draft.controller_step_id
            and draft.controller_step_id in candidate.member_step_ids
        )
        parent_for[draft.controller_step_id] = (
            min(
                candidates,
                key=lambda item: (
                    len(item.member_step_ids),
                    step_indexes[item.controller_step_id],
                ),
            ).controller_step_id
            if candidates
            else None
        )

    depths: dict[UUID, int] = {}

    def resolve_depth(controller: UUID, stack: tuple[UUID, ...] = ()) -> int:
        if controller in depths:
            return depths[controller]
        if controller in stack:
            diagnostics.append(
                GraphDiagnostic(
                    code="invalid_control_nesting",
                    severity="error",
                    message="control region nesting is cyclic",
                    node_id=controller,
                )
            )
            depths[controller] = 0
            return 0
        parent = parent_for[controller]
        depth = (
            0 if parent is None else resolve_depth(parent, stack + (controller,)) + 1
        )
        depths[controller] = depth
        return depth

    for controller in drafts_by_controller:
        resolve_depth(controller)

    regions = tuple(
        ControlRegion(
            kind=draft.kind,
            controller_step_id=draft.controller_step_id,
            parent_controller_step_id=parent_for[draft.controller_step_id],
            nesting_depth=depths[draft.controller_step_id],
            branches=draft.branches,
            iterable_step_ids=draft.iterable_step_ids,
            body_step_ids=draft.body_step_ids,
            boundary_step_ids=draft.boundary_step_ids,
            max_iterations=draft.max_iterations,
        )
        for draft in sorted(
            drafts, key=lambda item: step_indexes[item.controller_step_id]
        )
    )
    return regions, tuple(diagnostics)


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
        structural_diagnostics = validate_graph_structure(graph, resolved_context)
        if any(item.severity == "error" for item in structural_diagnostics):
            raise GraphCompilationError(structural_diagnostics)
        try:
            expansion = expand_subgraphs_with_trace(
                graph,
                resolved_context.subgraphs,
                max_depth=resolved_context.max_subgraph_depth,
            )
            expanded = expansion.graph
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
            definition = resolved_context.catalogue.get(
                node.type_id,
                node.type_version,
            )
            traits = definition.execution
            control = _control_annotation(node)
            cacheable = (
                is_constant
                and traits.pure
                and traits.deterministic
                and traits.cache_policy == "content"
                and not traits.side_effects
                and control is None
                and not resources.network
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
                    cacheable=cacheable,
                    critical_path_length=path_lengths[node_id],
                    on_critical_path=node_id in critical_nodes,
                    resources=resources,
                    control=control,
                    member_node_ids=(node_id,),
                )
            )

        collapsed_steps, execution_layers = _collapse_fused_steps(
            expanded,
            order,
            tuple(steps),
            resolved_context.catalogue,
        )
        steps = list(collapsed_steps)
        control_regions, control_diagnostics = _derive_control_regions(
            expanded,
            tuple(steps),
        )
        if control_diagnostics:
            diagnostics = tuple(
                sorted(
                    diagnostics + control_diagnostics,
                    key=lambda item: (
                        0 if item.severity == "error" else 1,
                        item.code,
                        str(item.node_id) if item.node_id is not None else "",
                    ),
                )
            )
        if any(item.severity == "error" for item in control_diagnostics):
            raise GraphCompilationError(diagnostics)
        group_indexes = {
            node_id: group_index
            for group_index, layer in enumerate(execution_layers)
            for node_id in layer
        }
        steps_by_id = {step.node_id: step for step in steps}
        groups: list[ExecutionGroup] = []
        for group_index, layer in enumerate(execution_layers):
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
                "plan_version": 2,
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
            fusions=tuple(step.fusion for step in steps if step.fusion is not None),
            control_regions=control_regions,
            subgraph_traces=expansion.traces,
        )
