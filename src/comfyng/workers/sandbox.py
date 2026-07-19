from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec


class SandboxViolation(PermissionError):
    """Raised inside a worker when its declared permissions are exceeded."""


class SandboxPolicy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    allow_network: bool = False
    allow_subprocess: bool = False
    inherit_environment: bool = False
    environment: Mapping[str, str] = msgspec.field(default_factory=dict)
    working_directory: str | None = None
    filesystem_read_roots: tuple[str, ...] = ()
    filesystem_write_roots: tuple[str, ...] = ()
    allow_runtime_imports: bool = True
    umask: int = 0o077
    max_open_files: int | None = None
    max_processes: int | None = None
    max_address_space_bytes: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.umask <= 0o777:
            raise ValueError("umask must be between 0 and 0o777")
        for field, value in (
            ("max_open_files", self.max_open_files),
            ("max_processes", self.max_processes),
            ("max_address_space_bytes", self.max_address_space_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be positive")
        if self.working_directory is not None and not self.working_directory:
            raise ValueError("working_directory must not be empty")
        for key, value in self.environment.items():
            if not key or "=" in key or not isinstance(value, str):
                raise ValueError(
                    "environment must contain valid string key/value pairs"
                )
        for field in ("filesystem_read_roots", "filesystem_write_roots"):
            raw_roots = getattr(self, field)
            if not isinstance(raw_roots, tuple):
                raise ValueError(f"{field} must be a tuple")
            resolved: list[str] = []
            for raw_root in raw_roots:
                if not isinstance(raw_root, str) or not raw_root:
                    raise ValueError(f"{field} must contain non-empty paths")
                root = Path(raw_root).expanduser().resolve(strict=True)
                if not root.is_dir():
                    raise ValueError(f"{field} roots must be directories")
                resolved.append(str(root))
            if len(set(resolved)) != len(resolved):
                raise ValueError(f"{field} must not contain duplicate roots")
            object.__setattr__(self, field, tuple(sorted(resolved)))

    @classmethod
    def from_permissions(
        cls,
        permissions: Mapping[str, object] | object,
        *,
        roots: Mapping[str, Path | str],
        working_directory: Path | str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> SandboxPolicy:
        """Convert a PermissionSet-like value without importing plugin modules."""

        if not isinstance(permissions, Mapping):
            converter = getattr(permissions, "to_mapping", None)
            if not callable(converter):
                raise TypeError("permissions must be a mapping or expose to_mapping()")
            permissions = converter()
        if not isinstance(permissions, Mapping):
            raise TypeError("permission conversion must return a mapping")

        def boolean(name: str) -> bool:
            value = permissions.get(name, False)
            if not isinstance(value, bool):
                raise TypeError(f"permission {name} must be a boolean")
            return value

        def aliases(name: str) -> tuple[str, ...]:
            value = permissions.get(name, ())
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(alias, str) and alias for alias in value
            ):
                raise TypeError(f"permission {name} must be a sequence of aliases")
            return tuple(value)

        resolved_roots: dict[str, str] = {}
        for alias, raw_root in roots.items():
            if not isinstance(alias, str) or not alias:
                raise ValueError("permission root aliases must not be empty")
            root = Path(raw_root).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError(f"permission root {alias!r} must be a directory")
            resolved_roots[alias] = str(root)
        read_aliases = aliases("filesystem_read")
        write_aliases = aliases("filesystem_write")
        missing = set((*read_aliases, *write_aliases)) - set(resolved_roots)
        if missing:
            raise ValueError(
                f"permission roots are missing: {', '.join(sorted(missing))}"
            )
        return cls(
            allow_network=boolean("network"),
            allow_subprocess=boolean("subprocess"),
            inherit_environment=False,
            environment={} if environment is None else dict(environment),
            working_directory=(
                None if working_directory is None else str(Path(working_directory))
            ),
            filesystem_read_roots=tuple(
                resolved_roots[alias] for alias in read_aliases
            ),
            filesystem_write_roots=tuple(
                resolved_roots[alias] for alias in write_aliases
            ),
        )


_SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
)


def default_policy_for(kind: str) -> SandboxPolicy:
    restricted = kind == "plugin"
    return SandboxPolicy(
        allow_network=not restricted,
        allow_subprocess=not restricted,
        inherit_environment=not restricted,
    )


def _lower_resource_limit(resource_id: int, requested: int) -> None:
    import resource

    soft, hard = resource.getrlimit(resource_id)
    target = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    if soft != resource.RLIM_INFINITY:
        target = min(target, soft)
    resource.setrlimit(resource_id, (target, hard))


def _resolve_audited_path(value: object, *, strict: bool) -> Path:
    if isinstance(value, int):
        raise TypeError("file-descriptor access is already bound to an open handle")
    try:
        path = Path(os.fsdecode(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SandboxViolation("filesystem access denied for an invalid path") from exc
    try:
        return path.expanduser().resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        access = "read" if strict else "write"
        raise SandboxViolation(f"filesystem {access} denied: {path}") from exc


def _is_within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _install_audit_guard(
    *,
    allow_network: bool,
    allow_subprocess: bool,
    read_roots: tuple[Path, ...],
    write_roots: tuple[Path, ...],
) -> None:
    network_events = frozenset({"socket.bind", "socket.connect", "socket.getaddrinfo"})
    process_prefixes = (
        "subprocess.",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "os.fork",
        "os.exec",
        "pty.spawn",
    )
    null_device = Path(os.devnull).resolve(strict=True)

    def require_path(value: object, *, write: bool) -> None:
        if isinstance(value, int):
            return
        path = _resolve_audited_path(value, strict=not write)
        if path == null_device:
            return
        roots = write_roots if write else read_roots
        if not _is_within(path, roots):
            access = "write" if write else "read"
            raise SandboxViolation(f"filesystem {access} denied: {path}")

    def open_is_write(arguments: tuple[Any, ...]) -> bool:
        mode = arguments[1] if len(arguments) > 1 else None
        flags = arguments[2] if len(arguments) > 2 else 0
        if isinstance(mode, str) and any(marker in mode for marker in "wax+"):
            return True
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return isinstance(flags, int) and bool(flags & write_flags)

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if not allow_network and event in network_events:
            raise SandboxViolation("network access denied by worker sandbox")
        if not allow_subprocess and event.startswith(process_prefixes):
            raise SandboxViolation("subprocess creation denied by worker sandbox")
        if event == "open" and arguments:
            require_path(arguments[0], write=open_is_write(arguments))
        elif event in {"os.listdir", "os.scandir", "os.chdir"} and arguments:
            if arguments[0] is not None:
                require_path(arguments[0], write=False)
        elif (
            event
            in {
                "os.remove",
                "os.rmdir",
                "os.mkdir",
                "os.chmod",
                "os.chown",
                "os.truncate",
                "os.utime",
            }
            and arguments
        ):
            require_path(arguments[0], write=True)
        elif event in {"os.rename", "shutil.move"} and len(arguments) >= 2:
            require_path(arguments[0], write=True)
            require_path(arguments[1], write=True)
        elif (
            event in {"os.link", "shutil.copyfile", "shutil.copy2"}
            and len(arguments) >= 2
        ):
            require_path(arguments[0], write=False)
            require_path(arguments[1], write=True)
        elif event == "os.symlink" and len(arguments) >= 2:
            destination = _resolve_audited_path(arguments[1], strict=False)
            if not _is_within(destination, write_roots):
                raise SandboxViolation(f"filesystem write denied: {destination}")
            source = Path(os.fsdecode(arguments[0]))
            if not source.is_absolute():
                source = destination.parent / source
            resolved_source = source.resolve(strict=False)
            if not _is_within(resolved_source, write_roots):
                raise SandboxViolation(
                    f"filesystem symlink target denied: {resolved_source}"
                )

    sys.addaudithook(audit)


def apply_sandbox(
    policy: SandboxPolicy,
    *,
    environment_overrides: Mapping[str, str] | None = None,
) -> None:
    """Apply irreversible, process-local restrictions in a freshly spawned worker."""

    overrides = dict(environment_overrides or {})
    original_working_directory = Path.cwd().resolve(strict=True)
    runtime_roots: set[Path] = set()
    if policy.allow_runtime_imports:
        runtime_roots.add(original_working_directory)
        for raw_path in sys.path:
            candidate = Path(raw_path or original_working_directory)
            try:
                resolved = candidate.expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved.is_dir():
                runtime_roots.add(resolved)
    if not policy.inherit_environment:
        retained = {
            key: value
            for key, value in os.environ.items()
            if key in _SAFE_ENVIRONMENT_KEYS
        }
        os.environ.clear()
        os.environ.update(retained)
    os.environ.update(policy.environment)
    os.environ.update(overrides)
    os.umask(policy.umask)
    if policy.working_directory is not None:
        directory = Path(policy.working_directory).expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("sandbox working_directory must be a directory")
        os.chdir(directory)
        runtime_roots.add(directory)
    if os.name == "posix":
        import resource

        if policy.max_open_files is not None:
            _lower_resource_limit(resource.RLIMIT_NOFILE, policy.max_open_files)
        if policy.max_processes is not None and hasattr(resource, "RLIMIT_NPROC"):
            _lower_resource_limit(resource.RLIMIT_NPROC, policy.max_processes)
        if policy.max_address_space_bytes is not None and hasattr(
            resource, "RLIMIT_AS"
        ):
            _lower_resource_limit(resource.RLIMIT_AS, policy.max_address_space_bytes)
    sys.dont_write_bytecode = True
    read_roots = tuple(
        sorted(
            runtime_roots.union(map(Path, policy.filesystem_read_roots)),
            key=str,
        )
    )
    write_roots = tuple(map(Path, policy.filesystem_write_roots))
    _install_audit_guard(
        allow_network=policy.allow_network,
        allow_subprocess=policy.allow_subprocess,
        read_roots=read_roots,
        write_roots=write_roots,
    )
