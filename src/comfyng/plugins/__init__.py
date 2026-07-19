"""Declarative, installable and process-isolated plugin APIs."""

from .catalogue import NodeCatalogue
from .environments import EnvironmentError, EnvironmentManager
from .installer import (
    InMemoryPluginRegistry,
    InstalledPlugin,
    InstallPhase,
    PluginInstallError,
    PluginInstaller,
    bundle_digest,
)
from .lifecycle import (
    PluginBusyError,
    PluginRuntimeError,
    PluginRuntimeManager,
    RuntimePluginRecord,
    RuntimePluginSpec,
    TrustGroupPermissionMismatch,
    UnknownPluginError,
)
from .manifest import (
    NodeDefinition,
    NodeExecutionTraits,
    PackageMetadata,
    PluginManifest,
    ResourceRequirements,
    RuntimeDefinition,
    load_json_schema,
)
from .permissions import PermissionDenied, PermissionGuard, PermissionSet
from .signatures import (
    HMACSignatureVerifier,
    SignatureVerificationError,
    SignatureVerifier,
)
from .worker import PluginMultiplexerRuntime, PluginWorkerError, SupervisorPluginBackend

__all__ = [
    "EnvironmentError",
    "EnvironmentManager",
    "HMACSignatureVerifier",
    "InMemoryPluginRegistry",
    "InstalledPlugin",
    "InstallPhase",
    "NodeCatalogue",
    "NodeDefinition",
    "NodeExecutionTraits",
    "PackageMetadata",
    "PermissionDenied",
    "PermissionGuard",
    "PermissionSet",
    "PluginBusyError",
    "PluginInstallError",
    "PluginInstaller",
    "PluginManifest",
    "PluginMultiplexerRuntime",
    "PluginRuntimeError",
    "PluginRuntimeManager",
    "PluginWorkerError",
    "ResourceRequirements",
    "RuntimePluginRecord",
    "RuntimePluginSpec",
    "RuntimeDefinition",
    "SignatureVerificationError",
    "SignatureVerifier",
    "SupervisorPluginBackend",
    "TrustGroupPermissionMismatch",
    "UnknownPluginError",
    "bundle_digest",
    "load_json_schema",
]
