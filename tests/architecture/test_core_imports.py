from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib


PROJECT_ROOT = Path(__file__).parents[2]
FORBIDDEN_DISTRIBUTIONS = {
    "civitai",
    "diffusers",
    "huggingface-hub",
    "torch",
    "transformers",
}
FORBIDDEN_MODULES = ("civitai", "diffusers", "huggingface_hub", "torch", "transformers")


def test_python_floor_and_reproducible_version_are_declared() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.14"
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.14.6"


def test_core_dependencies_exclude_ml_and_provider_libraries() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        dependency.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
        for dependency in pyproject["project"]["dependencies"]
    }

    assert {"fastapi", "pydantic", "pydantic-settings", "pyyaml", "typer"} <= dependencies
    assert dependencies.isdisjoint(FORBIDDEN_DISTRIBUTIONS)


def test_importing_core_does_not_import_ml_or_provider_modules() -> None:
    script = f"""
import json
import sys
import comfyng

forbidden = {FORBIDDEN_MODULES!r}
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in forbidden)
)
print(json.dumps(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_cli_help_lists_every_required_command_group() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "comfyng.cli.main", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for command in ("serve", "doctor", "benchmark", "models", "plugins", "jobs", "cache", "workers"):
        assert command in result.stdout


def test_cli_help_lists_every_required_subcommand() -> None:
    expected = {
        "models": ("list", "inspect", "import", "download"),
        "plugins": ("list", "install", "disable"),
        "jobs": ("list", "cancel"),
        "cache": ("inspect", "clean"),
        "workers": ("status",),
    }

    for group, commands in expected.items():
        result = subprocess.run(
            [sys.executable, "-m", "comfyng.cli.main", group, "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        for command in commands:
            assert command in result.stdout
