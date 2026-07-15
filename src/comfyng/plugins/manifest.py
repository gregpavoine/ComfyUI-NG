from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import tomllib
from typing import Any, ClassVar

import msgspec
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from comfyng.core.contracts import Contract, register_contract
from comfyng.core.enums import (
    GpuRequirement,
    LoadPolicy,
    RuntimeIsolation,
    UnloadPolicy,
)
from comfyng.core.errors import ManifestValidationError, PathContainmentError
from comfyng.core.ids import (
    validate_entrypoint,
    validate_node_id,
    validate_package_id,
    validate_semver,
)
from comfyng.core.json_values import validate_json_value
from comfyng.graph.types import DEFAULT_TYPE_REGISTRY, TypeRef


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ManifestValidationError(f"{field} must be a boolean")
    return value


def _table(
    value: object,
    *,
    context: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{context} must be a TOML table")
    table = dict(value)
    missing = required - set(table)
    unknown = set(table) - required - optional
    if missing:
        raise ManifestValidationError(
            f"{context} is missing required fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ManifestValidationError(
            f"{context} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    return table


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError(f"JSON schema contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_type_references(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        port_type = value.get("x-comfyng-type")
        if port_type is not None:
            try:
                DEFAULT_TYPE_REGISTRY.resolve_ref(TypeRef.parse(port_type))
            except (TypeError, ValueError) as exc:
                raise ManifestValidationError(
                    f"schema {location} has invalid x-comfyng-type: {port_type!r}"
                ) from exc
        for key, child in value.items():
            if key != "x-comfyng-type":
                _validate_type_references(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_type_references(child, location=f"{location}[{index}]")


def load_json_schema(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        schema = json.loads(source, object_pairs_hook=_unique_object)
    except ManifestValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"invalid JSON schema {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise ManifestValidationError(f"schema {path} must be a JSON object")
    if schema.get("$schema") != JSON_SCHEMA_DIALECT:
        raise ManifestValidationError(
            f"schema {path} must declare {JSON_SCHEMA_DIALECT!r}"
        )
    if schema.get("type") != "object":
        raise ManifestValidationError(f"schema {path} root type must be object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ManifestValidationError(
            f"schema {path} is not valid Draft 2020-12 JSON Schema: {exc.message}"
        ) from exc
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ManifestValidationError(f"schema {path}.properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) for name in required
    ):
        raise ManifestValidationError(f"schema {path}.required must be a string array")
    if len(required) != len(set(required)):
        raise ManifestValidationError(f"schema {path}.required contains duplicates")
    missing = set(required) - set(properties)
    if missing:
        raise ManifestValidationError(
            f"schema {path}.required references unknown properties: "
            f"{', '.join(sorted(missing))}"
        )
    _validate_type_references(schema, location=str(path))
    return schema


def _resolve_schema_path(
    raw_path: object,
    *,
    manifest_path: Path,
    root: Path,
) -> Path:
    value = _string(raw_path, field="node schema path")
    relative = Path(value)
    if relative.is_absolute():
        raise PathContainmentError("node schema paths must be relative")
    try:
        resolved = (manifest_path.parent / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ManifestValidationError(f"node schema does not exist: {value}") from exc
    if not resolved.is_relative_to(root):
        raise PathContainmentError(
            f"node schema path escapes catalogue root: {value}"
        )
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ManifestValidationError(f"node schema must be a JSON file: {value}")
    return resolved


@register_contract
class PackageMetadata(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.plugin-package"

    id: str
    name: str
    version: str
    publisher: str
    license: str
    description: str | None = None

    def __post_init__(self) -> None:
        validate_package_id(self.id)
        _string(self.name, field="package.name")
        validate_semver(self.version, field="package.version")
        _string(self.publisher, field="package.publisher")
        _string(self.license, field="package.license")
        if self.description is not None:
            _string(self.description, field="package.description")


@register_contract
class RuntimeDefinition(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.plugin-runtime"

    language: str
    python: str
    entrypoint: str
    isolation: RuntimeIsolation
    load_policy: LoadPolicy
    unload_policy: UnloadPolicy
    idle_timeout_seconds: int

    def __post_init__(self) -> None:
        if self.language != "python":
            raise ValueError("runtime.language must be 'python'")
        if self.python != ">=3.14":
            raise ValueError("runtime.python must be '>=3.14'")
        validate_entrypoint(self.entrypoint)
        if not isinstance(self.isolation, RuntimeIsolation):
            raise ValueError("runtime.isolation is invalid")
        if not isinstance(self.load_policy, LoadPolicy):
            raise ValueError("runtime.load_policy is invalid")
        if not isinstance(self.unload_policy, UnloadPolicy):
            raise ValueError("runtime.unload_policy is invalid")
        if type(self.idle_timeout_seconds) is not int or self.idle_timeout_seconds < 0:
            raise ValueError("runtime.idle_timeout_seconds must be non-negative")


@register_contract
class ResourceRequirements(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.plugin-resources"

    gpu: GpuRequirement
    estimated_ram_mb: int
    estimated_vram_mb: int
    network: bool

    def __post_init__(self) -> None:
        if not isinstance(self.gpu, GpuRequirement):
            raise ValueError("resources.gpu is invalid")
        if type(self.estimated_ram_mb) is not int or self.estimated_ram_mb < 0:
            raise ValueError("resources.estimated_ram_mb must be non-negative")
        if type(self.estimated_vram_mb) is not int or self.estimated_vram_mb < 0:
            raise ValueError("resources.estimated_vram_mb must be non-negative")
        if type(self.network) is not bool:
            raise ValueError("resources.network must be a boolean")


@register_contract
class NodeDefinition(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.node-definition"

    id: str
    version: str
    display_name: str
    package_id: str
    input_schema_path: Path
    output_schema_path: Path
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    category: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        validate_node_id(self.id)
        validate_semver(self.version, field="node.version")
        _string(self.display_name, field="node.display_name")
        validate_package_id(self.package_id)
        for field in ("input_schema_path", "output_schema_path"):
            value = getattr(self, field)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field} must be an absolute path")
        for field in ("input_schema", "output_schema"):
            value = getattr(self, field)
            if type(value) is not dict:
                raise ValueError(f"{field} must be a JSON object")
            validate_json_value(value, path=f"$.{field}")
        for field in ("category", "description"):
            value = getattr(self, field)
            if value is not None:
                _string(value, field=f"node.{field}")


@register_contract
class PluginManifest(Contract):
    TYPE_ID: ClassVar[str] = "comfyng.plugin-manifest"

    schema_version: int
    package: PackageMetadata
    runtime: RuntimeDefinition
    resources: ResourceRequirements
    nodes: tuple[NodeDefinition, ...]
    source_path: Path
    permissions: Mapping[str, bool] = msgspec.field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    signature: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("manifest schema_version must be 1")
        if not isinstance(self.package, PackageMetadata):
            raise ValueError("manifest package is invalid")
        if not isinstance(self.runtime, RuntimeDefinition):
            raise ValueError("manifest runtime is invalid")
        if not isinstance(self.resources, ResourceRequirements):
            raise ValueError("manifest resources are invalid")
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise ValueError("manifest nodes must be a non-empty tuple")
        if any(not isinstance(node, NodeDefinition) for node in self.nodes):
            raise ValueError("manifest nodes must contain NodeDefinition values")
        node_keys = {(node.id, node.version) for node in self.nodes}
        if len(node_keys) != len(self.nodes):
            raise ValueError("manifest contains duplicate node id/version pairs")
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise ValueError("manifest source_path must be absolute")
        if not isinstance(self.permissions, Mapping) or any(
            not isinstance(name, str) or type(enabled) is not bool
            for name, enabled in self.permissions.items()
        ):
            raise ValueError("manifest permissions must be a boolean mapping")
        if any(not isinstance(item, str) or not item for item in self.dependencies):
            raise ValueError("manifest dependencies must be non-empty strings")
        if self.signature is not None:
            _string(self.signature, field="manifest.signature")

    @classmethod
    def load(cls, path: Path | str, *, root: Path | str | None = None) -> PluginManifest:
        raw_path = Path(path)
        try:
            source_path = raw_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ManifestValidationError(f"manifest does not exist: {raw_path}") from exc
        if not source_path.is_file() or source_path.name != "ng-node.toml":
            raise ManifestValidationError("manifest path must name an ng-node.toml file")
        raw_root = Path(root) if root is not None else source_path.parent
        try:
            resolved_root = raw_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ManifestValidationError(f"catalogue root does not exist: {raw_root}") from exc
        if not resolved_root.is_dir():
            raise ManifestValidationError("catalogue root must be a directory")
        if not source_path.is_relative_to(resolved_root):
            raise PathContainmentError("manifest path escapes catalogue root")
        try:
            payload = tomllib.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ManifestValidationError(f"invalid TOML manifest {source_path}: {exc}") from exc
        return _parse_manifest(payload, path=source_path, root=resolved_root)


def _parse_manifest(
    payload: Mapping[str, Any],
    *,
    path: Path,
    root: Path,
) -> PluginManifest:
    top = _table(
        payload,
        context="manifest",
        required=frozenset(("schema_version", "package", "runtime", "resources", "nodes")),
        optional=frozenset(("permissions", "dependencies", "signature")),
    )
    schema_version = _integer(
        top["schema_version"], field="manifest.schema_version", minimum=1
    )
    if schema_version != 1:
        raise ManifestValidationError(
            f"unsupported manifest schema_version: {schema_version}"
        )

    package_raw = _table(
        top["package"],
        context="package",
        required=frozenset(("id", "name", "version", "publisher", "license")),
        optional=frozenset(("description",)),
    )
    try:
        package = PackageMetadata(
            id=validate_package_id(package_raw["id"]),
            name=_string(package_raw["name"], field="package.name"),
            version=validate_semver(package_raw["version"], field="package.version"),
            publisher=_string(package_raw["publisher"], field="package.publisher"),
            license=_string(package_raw["license"], field="package.license"),
            description=package_raw.get("description"),
        )
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc

    runtime_raw = _table(
        top["runtime"],
        context="runtime",
        required=frozenset(
            (
                "language",
                "python",
                "entrypoint",
                "isolation",
                "unload_policy",
                "idle_timeout_seconds",
            )
        ),
        optional=frozenset(("load_policy",)),
    )
    try:
        runtime = RuntimeDefinition(
            language=_string(runtime_raw["language"], field="runtime.language"),
            python=_string(runtime_raw["python"], field="runtime.python"),
            entrypoint=validate_entrypoint(runtime_raw["entrypoint"]),
            isolation=RuntimeIsolation(runtime_raw["isolation"]),
            load_policy=LoadPolicy(
                runtime_raw.get("load_policy", LoadPolicy.LOAD_ON_EXECUTION.value)
            ),
            unload_policy=UnloadPolicy(runtime_raw["unload_policy"]),
            idle_timeout_seconds=_integer(
                runtime_raw["idle_timeout_seconds"],
                field="runtime.idle_timeout_seconds",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"invalid runtime metadata: {exc}") from exc

    resources_raw = _table(
        top["resources"],
        context="resources",
        required=frozenset(
            ("gpu", "estimated_ram_mb", "estimated_vram_mb", "network")
        ),
    )
    try:
        resources = ResourceRequirements(
            gpu=GpuRequirement(resources_raw["gpu"]),
            estimated_ram_mb=_integer(
                resources_raw["estimated_ram_mb"], field="resources.estimated_ram_mb"
            ),
            estimated_vram_mb=_integer(
                resources_raw["estimated_vram_mb"],
                field="resources.estimated_vram_mb",
            ),
            network=_boolean(resources_raw["network"], field="resources.network"),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"invalid resource metadata: {exc}") from exc

    nodes_raw = top["nodes"]
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ManifestValidationError("manifest.nodes must be a non-empty array of tables")
    nodes: list[NodeDefinition] = []
    seen_nodes: set[tuple[str, str]] = set()
    for index, raw_node in enumerate(nodes_raw):
        node_raw = _table(
            raw_node,
            context=f"nodes[{index}]",
            required=frozenset(("id", "display_name", "input_schema", "output_schema")),
            optional=frozenset(("version", "category", "description")),
        )
        try:
            node_id = validate_node_id(node_raw["id"])
            version = validate_semver(
                node_raw.get("version", package.version), field="node.version"
            )
            key = (node_id, version)
            if key in seen_nodes:
                raise ManifestValidationError(
                    f"duplicate node definition {node_id}@{version} in {path}"
                )
            seen_nodes.add(key)
            input_path = _resolve_schema_path(
                node_raw["input_schema"], manifest_path=path, root=root
            )
            output_path = _resolve_schema_path(
                node_raw["output_schema"], manifest_path=path, root=root
            )
            nodes.append(
                NodeDefinition(
                    id=node_id,
                    version=version,
                    display_name=_string(
                        node_raw["display_name"], field="node.display_name"
                    ),
                    package_id=package.id,
                    input_schema_path=input_path,
                    output_schema_path=output_path,
                    input_schema=load_json_schema(input_path),
                    output_schema=load_json_schema(output_path),
                    category=node_raw.get("category"),
                    description=node_raw.get("description"),
                )
            )
        except (ManifestValidationError, PathContainmentError):
            raise
        except ValueError as exc:
            raise ManifestValidationError(f"invalid nodes[{index}]: {exc}") from exc

    permissions_raw = top.get("permissions", {})
    if not isinstance(permissions_raw, Mapping) or any(
        not isinstance(name, str) or type(enabled) is not bool
        for name, enabled in permissions_raw.items()
    ):
        raise ManifestValidationError("permissions must be a boolean TOML table")
    dependencies_raw = top.get("dependencies", [])
    if not isinstance(dependencies_raw, list) or any(
        not isinstance(item, str) or not item for item in dependencies_raw
    ):
        raise ManifestValidationError("dependencies must be an array of strings")
    signature = top.get("signature")
    if signature is not None:
        signature = _string(signature, field="manifest.signature")
    try:
        return PluginManifest(
            schema_version=schema_version,
            package=package,
            runtime=runtime,
            resources=resources,
            nodes=tuple(nodes),
            source_path=path,
            permissions=dict(permissions_raw),
            dependencies=tuple(dependencies_raw),
            signature=signature,
        )
    except ValueError as exc:
        raise ManifestValidationError(str(exc)) from exc
