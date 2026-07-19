from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
import tomllib
from types import MappingProxyType
from typing import Protocol, TypeVar
import zipfile

from comfyng.plugins.environments import EnvironmentManager
from comfyng.plugins.manifest import PluginManifest
from comfyng.plugins.permissions import PermissionSet
from comfyng.plugins.signatures import SignatureVerifier


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ResultT = TypeVar("_ResultT")


class InstallPhase(StrEnum):
    DOWNLOAD = "download"
    VERIFY_SIGNATURE = "verify_signature"
    VERIFY_HASH = "verify_hash"
    READ_PERMISSIONS = "read_permissions"
    RESOLVE_DEPENDENCIES = "resolve_dependencies"
    CREATE_ENVIRONMENT = "create_environment"
    IMPORT_TEST = "import_test"
    VALIDATE_MANIFEST = "validate_manifest"
    PUBLISH = "publish"
    REGISTER = "register"


class PluginInstallError(RuntimeError):
    def __init__(self, phase: InstallPhase, message: str) -> None:
        self.phase = phase
        super().__init__(f"plugin installation failed during {phase.value}: {message}")


@dataclass(frozen=True, slots=True)
class InstalledPlugin:
    package_id: str
    version: str
    digest: str
    path: Path
    manifest: PluginManifest
    permissions: PermissionSet
    installed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("installed plugin path must be absolute")
        if _SHA256.fullmatch(self.digest) is None:
            raise ValueError("installed plugin digest must be SHA-256")
        if self.package_id != self.manifest.package.id:
            raise ValueError("installed package id must match its manifest")
        if self.version != self.manifest.package.version:
            raise ValueError("installed version must match its manifest")
        if not isinstance(self.permissions, PermissionSet):
            raise ValueError("installed permissions must be a PermissionSet")
        if self.installed_at.tzinfo is None:
            raise ValueError("installed_at must be timezone-aware")


class PluginRegistry(Protocol):
    def register(self, plugin: InstalledPlugin) -> None: ...

    def unregister(self, plugin: InstalledPlugin) -> None: ...


class InMemoryPluginRegistry:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], InstalledPlugin] = {}
        self._lock = threading.RLock()

    @property
    def records(self) -> Mapping[tuple[str, str], InstalledPlugin]:
        with self._lock:
            return MappingProxyType(dict(self._records))

    def register(self, plugin: InstalledPlugin) -> None:
        key = (plugin.package_id, plugin.version)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None and existing.digest != plugin.digest:
                raise ValueError(f"plugin {key[0]}@{key[1]} is already registered")
            self._records[key] = plugin

    def unregister(self, plugin: InstalledPlugin) -> None:
        key = (plugin.package_id, plugin.version)
        with self._lock:
            if self._records.get(key) is plugin:
                self._records.pop(key, None)


def bundle_digest(root: Path | str) -> str:
    path = Path(root).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("plugin bundle must be a directory")
    digest = sha256()
    files: list[Path] = []
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("plugin bundles cannot contain symbolic links")
        if candidate.is_file():
            files.append(candidate)
        elif not candidate.is_dir():
            raise ValueError("plugin bundles cannot contain special files")
    for candidate in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = candidate.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("plugin archive contains a path traversal")
            mode = member.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise ValueError("plugin archive contains a symbolic link")
            target = (destination / Path(*relative.parts)).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("plugin archive escapes the staging directory")
        archive.extractall(destination)


def _materialize(source: Path, destination: Path) -> None:
    resolved = source.resolve(strict=True)
    if resolved.is_dir():
        for candidate in resolved.rglob("*"):
            if candidate.is_symlink():
                raise ValueError("plugin source cannot contain symbolic links")
        shutil.copytree(resolved, destination)
        return
    if resolved.is_file() and resolved.suffix.casefold() == ".zip":
        destination.mkdir()
        _safe_extract_zip(resolved, destination)
        children = tuple(destination.iterdir())
        if len(children) == 1 and children[0].is_dir():
            wrapper = children[0]
            for child in tuple(wrapper.iterdir()):
                os.replace(child, destination / child.name)
            wrapper.rmdir()
        return
    raise ValueError("plugin source must be a directory or ZIP archive")


def _raw_manifest(bundle: Path) -> dict[str, object]:
    path = bundle / "ng-node.toml"
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read ng-node.toml: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("ng-node.toml must contain a TOML table")
    return value


def _table(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest {name} must be a table")
    return dict(value)


class PluginInstaller:
    def __init__(
        self,
        root: Path | str,
        *,
        registry: PluginRegistry,
        signature_verifier: SignatureVerifier,
        environment_manager: EnvironmentManager | None = None,
        phase_observer: Callable[[InstallPhase], None] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.staging_root = self.root / "staging"
        self.installed_root = self.root / "installed"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.installed_root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.signature_verifier = signature_verifier
        self.environment_manager = environment_manager or EnvironmentManager()
        self.phase_observer = phase_observer
        self._publish_lock = threading.RLock()

    def _phase(
        self,
        phase: InstallPhase,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        try:
            result = operation()
            if self.phase_observer is not None:
                self.phase_observer(phase)
            return result
        except PluginInstallError:
            raise
        except Exception as exc:
            raise PluginInstallError(phase, str(exc)) from exc

    def install(
        self,
        source: Path | str,
        *,
        expected_sha256: str,
        signature: str,
    ) -> InstalledPlugin:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise PluginInstallError(
                InstallPhase.VERIFY_HASH,
                "expected_sha256 must be lowercase SHA-256",
            )
        staging = Path(tempfile.mkdtemp(prefix="install-", dir=self.staging_root))
        bundle = staging / "bundle"
        published_path: Path | None = None
        registered: InstalledPlugin | None = None
        try:
            self._phase(
                InstallPhase.DOWNLOAD,
                lambda: _materialize(Path(source), bundle),
            )

            def verify_signature() -> dict[str, object]:
                raw_manifest = _raw_manifest(bundle)
                package_raw = _table(raw_manifest.get("package"), "package")
                publisher = package_raw.get("publisher")
                if not isinstance(publisher, str) or not publisher:
                    raise ValueError("manifest package.publisher is required")
                self.signature_verifier.verify(
                    publisher,
                    expected_sha256,
                    signature,
                )
                return raw_manifest

            raw = self._phase(InstallPhase.VERIFY_SIGNATURE, verify_signature)
            actual_digest = self._phase(
                InstallPhase.VERIFY_HASH,
                lambda: bundle_digest(bundle),
            )
            if actual_digest != expected_sha256:
                raise PluginInstallError(
                    InstallPhase.VERIFY_HASH,
                    f"bundle hash mismatch: expected {expected_sha256}, got {actual_digest}",
                )

            def read_permissions() -> PermissionSet:
                permissions = PermissionSet.from_mapping(
                    _table(raw.get("permissions", {}), "permissions")
                )
                resources = _table(raw.get("resources"), "resources")
                if resources.get("network") is True and not permissions.network:
                    raise ValueError("network resources require network permission")
                if resources.get("gpu") == "required" and not permissions.gpu:
                    raise ValueError("required GPU resources require gpu permission")
                return permissions

            permissions = self._phase(
                InstallPhase.READ_PERMISSIONS,
                read_permissions,
            )
            dependencies = raw.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str) for item in dependencies
            ):
                raise PluginInstallError(
                    InstallPhase.RESOLVE_DEPENDENCIES,
                    "manifest dependencies must be an array of strings",
                )
            lockfile = self._phase(
                InstallPhase.RESOLVE_DEPENDENCIES,
                lambda: self.environment_manager.resolve(bundle, dependencies),
            )
            environment = self._phase(
                InstallPhase.CREATE_ENVIRONMENT,
                lambda: self.environment_manager.create(bundle, lockfile),
            )
            runtime_raw = _table(raw.get("runtime"), "runtime")
            entrypoint = runtime_raw.get("entrypoint")
            if not isinstance(entrypoint, str):
                raise PluginInstallError(
                    InstallPhase.IMPORT_TEST,
                    "manifest runtime.entrypoint is required",
                )
            self._phase(
                InstallPhase.IMPORT_TEST,
                lambda: self.environment_manager.test_import(
                    environment,
                    bundle,
                    entrypoint,
                ),
            )
            manifest = self._phase(
                InstallPhase.VALIDATE_MANIFEST,
                lambda: PluginManifest.load(bundle / "ng-node.toml", root=bundle),
            )
            destination = (
                self.installed_root / manifest.package.id / manifest.package.version
            ).resolve()
            if not destination.is_relative_to(self.installed_root.resolve()):
                raise PluginInstallError(
                    InstallPhase.PUBLISH,
                    "manifest package path escapes installed root",
                )

            def publish() -> Path:
                nonlocal published_path
                with self._publish_lock:
                    if destination.exists():
                        raise ValueError(
                            f"plugin {manifest.package.id}@{manifest.package.version} "
                            "is already installed"
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(bundle, destination)
                    published_path = destination
                return destination

            self._phase(InstallPhase.PUBLISH, publish)

            def register() -> InstalledPlugin:
                nonlocal registered
                installed_manifest = PluginManifest.load(
                    destination / "ng-node.toml",
                    root=destination,
                )
                result = InstalledPlugin(
                    package_id=installed_manifest.package.id,
                    version=installed_manifest.package.version,
                    digest=actual_digest,
                    path=destination,
                    manifest=installed_manifest,
                    permissions=permissions,
                    installed_at=datetime.now(UTC),
                )
                self.registry.register(result)
                registered = result
                return result

            return self._phase(InstallPhase.REGISTER, register)
        except Exception:
            if registered is not None:
                try:
                    self.registry.unregister(registered)
                except Exception:
                    pass
            if published_path is not None:
                shutil.rmtree(published_path, ignore_errors=True)
                try:
                    published_path.parent.rmdir()
                except OSError:
                    pass
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "InMemoryPluginRegistry",
    "InstallPhase",
    "InstalledPlugin",
    "PluginInstallError",
    "PluginInstaller",
    "PluginRegistry",
    "bundle_digest",
]
