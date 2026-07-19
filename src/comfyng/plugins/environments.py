from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import subprocess
import sys
import venv


class EnvironmentError(RuntimeError):
    pass


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


class EnvironmentManager:
    """Build a plugin-local Python environment without mutating the core one."""

    def resolve(self, bundle: Path, dependencies: Sequence[str]) -> Path:
        normalized: list[str] = []
        for dependency in dependencies:
            if not isinstance(dependency, str) or not dependency.strip():
                raise EnvironmentError("plugin dependencies must be non-empty strings")
            if any(character in dependency for character in ("\n", "\r", "\0")):
                raise EnvironmentError("plugin dependencies contain unsafe characters")
            normalized.append(dependency.strip())
        if len(set(normalized)) != len(normalized):
            raise EnvironmentError("plugin dependencies contain duplicates")
        lockfile = bundle / "lockfile"
        content = "".join(f"{item}\n" for item in sorted(normalized))
        try:
            lockfile.write_text(content, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EnvironmentError(f"cannot write plugin lockfile: {exc}") from exc
        return lockfile

    def create(self, bundle: Path, lockfile: Path) -> Path:
        environment = bundle / ".venv"
        if environment.exists():
            raise EnvironmentError("plugin bundle already contains a .venv")
        dependencies = tuple(
            line.strip()
            for line in lockfile.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        try:
            venv.EnvBuilder(
                with_pip=bool(dependencies),
                clear=False,
                symlinks=False,
            ).create(environment)
        except (OSError, subprocess.SubprocessError) as exc:
            raise EnvironmentError(f"cannot create plugin environment: {exc}") from exc
        if dependencies:
            command = (
                str(_environment_python(environment)),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--requirement",
                str(lockfile),
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=bundle,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise EnvironmentError(
                    f"dependency installation failed: {exc}"
                ) from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout)[-2000:]
                raise EnvironmentError(
                    f"dependency installation failed with exit {completed.returncode}: "
                    f"{detail}"
                )
        return environment

    def test_import(self, environment: Path, bundle: Path, entrypoint: str) -> None:
        module_name, separator, attribute = entrypoint.partition(":")
        if not separator or not module_name or not attribute:
            raise EnvironmentError("plugin entrypoint is invalid")
        script = (
            "import importlib, sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "module = importlib.import_module(sys.argv[2]); "
            "target = getattr(module, sys.argv[3]); "
            "assert callable(target)"
        )
        command = (
            str(_environment_python(environment)),
            "-c",
            script,
            str(bundle / "package"),
            module_name,
            attribute,
        )
        environment_variables = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
        }
        if sys.platform == "win32":
            environment_variables["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            completed = subprocess.run(
                command,
                cwd=bundle,
                env=environment_variables,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise EnvironmentError(f"plugin import test failed: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise EnvironmentError(
                f"plugin import test failed with exit {completed.returncode}: {detail}"
            )


__all__ = ["EnvironmentError", "EnvironmentManager"]
