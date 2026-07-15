from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from comfyng.core.errors import (
    CatalogueDiscoveryError,
    DuplicateNodeDefinitionError,
    UnknownNodeDefinitionError,
)
from comfyng.core.ids import semver_sort_key

from .manifest import NodeDefinition, PluginManifest


def _default_catalogue_root() -> Path:
    packaged = Path(__file__).resolve().parents[1] / "catalogue"
    if (packaged / "runtimes").is_dir():
        return packaged
    source = Path(__file__).resolve().parents[3]
    if (source / "runtimes").is_dir():
        return source
    raise CatalogueDiscoveryError("bundled node catalogue is unavailable")


def _manifest_paths(root: Path) -> tuple[Path, tuple[Path, ...]]:
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CatalogueDiscoveryError(f"catalogue path does not exist: {root}") from exc
    if resolved.is_file():
        if resolved.name != "ng-node.toml":
            raise CatalogueDiscoveryError("catalogue file must be named ng-node.toml")
        return resolved.parent, (resolved,)
    if not resolved.is_dir():
        raise CatalogueDiscoveryError("catalogue root must be a directory")
    if resolved.name == "runtimes" and (resolved.parent / "schemas").is_dir():
        catalogue_root = resolved.parent
        scan_root = resolved
    elif (resolved / "runtimes").is_dir():
        catalogue_root = resolved
        scan_root = resolved / "runtimes"
    else:
        catalogue_root = resolved
        scan_root = resolved
    manifests = tuple(sorted(scan_root.rglob("ng-node.toml")))
    if not manifests:
        raise CatalogueDiscoveryError(f"no ng-node.toml manifests found under {resolved}")
    return catalogue_root, manifests


@dataclass(frozen=True, slots=True)
class NodeCatalogue:
    manifests: tuple[PluginManifest, ...]
    nodes: tuple[NodeDefinition, ...]
    _node_index: Mapping[tuple[str, str], NodeDefinition] = field(
        init=False, repr=False, compare=False
    )
    _versions: Mapping[str, tuple[NodeDefinition, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        node_index: dict[tuple[str, str], NodeDefinition] = {}
        versions: dict[str, list[NodeDefinition]] = {}
        for node in self.nodes:
            key = (node.id, node.version)
            if key in node_index:
                raise DuplicateNodeDefinitionError(
                    f"duplicate node definition {node.id}@{node.version}"
                )
            node_index[key] = node
            versions.setdefault(node.id, []).append(node)
        ordered_versions = {
            node_id: tuple(
                sorted(items, key=lambda item: semver_sort_key(item.version))
            )
            for node_id, items in versions.items()
        }
        object.__setattr__(self, "_node_index", MappingProxyType(node_index))
        object.__setattr__(self, "_versions", MappingProxyType(ordered_versions))

    @classmethod
    def discover(cls, root: Path | str | None = None) -> NodeCatalogue:
        requested = Path(root) if root is not None else _default_catalogue_root()
        catalogue_root, manifest_paths = _manifest_paths(requested)
        manifests = tuple(
            PluginManifest.load(path, root=catalogue_root)
            for path in manifest_paths
        )
        package_keys: set[tuple[str, str]] = set()
        node_keys: set[tuple[str, str]] = set()
        nodes: list[NodeDefinition] = []
        for manifest in manifests:
            package_key = (manifest.package.id, manifest.package.version)
            if package_key in package_keys:
                raise DuplicateNodeDefinitionError(
                    "duplicate package manifest "
                    f"{manifest.package.id}@{manifest.package.version}"
                )
            package_keys.add(package_key)
            for node in manifest.nodes:
                node_key = (node.id, node.version)
                if node_key in node_keys:
                    raise DuplicateNodeDefinitionError(
                        f"duplicate node definition {node.id}@{node.version}"
                    )
                node_keys.add(node_key)
                nodes.append(node)
        ordered_nodes = tuple(sorted(nodes, key=lambda node: (node.id, node.version)))
        return cls(manifests=manifests, nodes=ordered_nodes)

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(node.display_name for node in self.nodes)

    @property
    def definitions(self) -> tuple[NodeDefinition, ...]:
        return self.nodes

    def get(self, node_id: str, version: str | None = None) -> NodeDefinition:
        if version is not None:
            try:
                return self._node_index[(node_id, version)]
            except KeyError as exc:
                raise UnknownNodeDefinitionError(
                    f"unknown node definition {node_id}@{version}"
                ) from exc
        try:
            return self._versions[node_id][-1]
        except KeyError as exc:
            raise UnknownNodeDefinitionError(
                f"unknown node definition {node_id}"
            ) from exc
