from __future__ import annotations


class ComfyNGError(Exception):
    """Base class for stable ComfyUI-NG domain errors."""


class ContractValidationError(ComfyNGError, ValueError):
    """Raised when a serialized domain contract is invalid or unsupported."""


class JsonValueValidationError(ContractValidationError):
    """Raised when a value cannot make a stable JSON round trip."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class IdentifierValidationError(ComfyNGError, ValueError):
    """Raised when a stable identifier does not use the declared syntax."""


class InvalidLifecycleTransition(ComfyNGError, ValueError):
    """Raised when a runtime lifecycle transition is not in the state machine."""

    def __init__(self, source: object, target: object) -> None:
        self.source = source
        self.target = target
        super().__init__(f"invalid lifecycle transition: {source} -> {target}")


class TypeRegistryError(ComfyNGError, ValueError):
    """Base class for versioned graph type registry errors."""


class DuplicateTypeDefinitionError(TypeRegistryError):
    """Raised when the same type identifier and version is registered twice."""


class UnknownTypeDefinitionError(TypeRegistryError):
    """Raised when a requested type identifier or version is not registered."""


class ManifestError(ComfyNGError, ValueError):
    """Base class for declarative plugin manifest failures."""


class ManifestValidationError(ManifestError):
    """Raised when TOML metadata or a referenced JSON schema is invalid."""


class PathContainmentError(ManifestError):
    """Raised when a manifest or schema path escapes its catalogue root."""


class CatalogueDiscoveryError(ManifestError):
    """Raised when a node catalogue cannot be discovered deterministically."""


class DuplicateNodeDefinitionError(CatalogueDiscoveryError):
    """Raised for duplicate package or node identifier/version pairs."""


class UnknownNodeDefinitionError(CatalogueDiscoveryError, KeyError):
    """Raised when a catalogue lookup cannot resolve a node definition."""
