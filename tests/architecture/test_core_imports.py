from __future__ import annotations

from importlib.metadata import distribution
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


PROJECT_ROOT = Path(__file__).parents[2]
EXPECTED_CORE_DISTRIBUTIONS = {
    "aiosqlite",
    "fastapi",
    "jsonschema",
    "msgspec",
    "pydantic",
    "pydantic-settings",
    "pyyaml",
    "typer",
}
FORBIDDEN_DISTRIBUTIONS = {
    "civitai",
    "cuda",
    "cupy",
    "diffusers",
    "flash-attn",
    "huggingface-hub",
    "nvidia",
    "pycuda",
    "sageattention",
    "torch",
    "transformers",
    "triton",
    "xformers",
}
FORBIDDEN_MODULES = (
    "civitai",
    "cuda",
    "cupy",
    "diffusers",
    "flash_attn",
    "huggingface_hub",
    "nvidia",
    "pycuda",
    "sageattention",
    "torch",
    "transformers",
    "triton",
    "xformers",
)
UNAVAILABLE_COMMANDS = (
    (("serve",), "api"),
    (("benchmark",), "benchmark"),
    (("models", "list"), "models"),
    (("models", "inspect"), "models"),
    (("models", "import"), "models"),
    (("models", "download"), "providers"),
    (("plugins", "list"), "plugins"),
    (("plugins", "install"), "plugins"),
    (("plugins", "disable"), "plugins"),
    (("jobs", "list"), "jobs"),
    (("jobs", "cancel"), "jobs"),
    (("cache", "inspect"), "cache"),
    (("cache", "clean"), "cache"),
    (("workers", "status"), "workers"),
)
NON_MAPPING_YAML_VALUES = (
    ("null", "null"),
    ("false", "false"),
    ("zero", "0"),
    ("empty-list", "[]"),
    ("string", '"not-a-mapping"'),
    ("malformed-list-of-pairs", "[[[path], value]]"),
)


def _console_script() -> Path:
    installed_distribution = distribution("comfyui-ng")
    console_entries = [
        entry
        for entry in installed_distribution.entry_points
        if entry.group == "console_scripts" and entry.name == "comfyng"
    ]
    assert [(entry.name, entry.value) for entry in console_entries] == [
        ("comfyng", "comfyng.cli.main:main")
    ]
    assert console_entries[0].load().__name__ == "main"

    script = Path(sys.executable).with_name("comfyng")
    assert script.is_file()
    return script


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_console_script(), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_python_floor_and_reproducible_version_are_declared() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.14"
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.14.6"


def test_core_dependency_metadata_is_exact_and_excludes_heavy_libraries() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        canonicalize_name(Requirement(value).name)
        for value in pyproject["project"]["dependencies"]
    }
    installed = {
        canonicalize_name(requirement.name)
        for value in distribution("comfyui-ng").requires or []
        if "extra" not in str((requirement := Requirement(value)).marker)
    }

    assert declared == EXPECTED_CORE_DISTRIBUTIONS
    assert installed == EXPECTED_CORE_DISTRIBUTIONS
    assert declared & FORBIDDEN_DISTRIBUTIONS == set()
    assert installed & FORBIDDEN_DISTRIBUTIONS == set()


def test_importing_core_does_not_import_heavy_or_provider_modules() -> None:
    script = f"""
import json
import sys
import comfyng
import comfyng.core
import comfyng.graph
import comfyng.models
import comfyng.plugins

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


def test_installed_console_entry_point_lists_every_required_command_group() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0, result.stderr
    for command in ("serve", "doctor", "benchmark", "models", "plugins", "jobs", "cache", "workers"):
        assert re.search(rf"\b{re.escape(command)}\b", result.stdout)


def test_installed_console_entry_point_lists_every_required_subcommand() -> None:
    expected = {
        "models": ("list", "inspect", "import", "download"),
        "plugins": ("list", "install", "disable"),
        "jobs": ("list", "cancel"),
        "cache": ("inspect", "clean"),
        "workers": ("status",),
    }

    for group, commands in expected.items():
        result = _run_cli(group, "--help")
        assert result.returncode == 0, result.stderr
        for command in commands:
            assert re.search(rf"\b{re.escape(command)}\b", result.stdout)


@pytest.mark.parametrize(("arguments", "service"), UNAVAILABLE_COMMANDS)
def test_unwired_commands_report_structured_service_unavailability(
    arguments: tuple[str, ...],
    service: str,
) -> None:
    result = _run_cli(*arguments)

    assert result.returncode == 69
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "error": {
            "code": "SERVICE_UNAVAILABLE",
            "message": "Required service is not available in this installation.",
            "service": service,
        }
    }


def test_doctor_json_reports_python_and_configuration_health() -> None:
    result = _run_cli("doctor", "--json")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    diagnostic = json.loads(result.stdout)
    assert diagnostic["status"] == "ok"
    assert diagnostic["checks"]["python"]["ok"] is True
    assert diagnostic["checks"]["python"]["required"] == ">=3.14"
    assert diagnostic["checks"]["configuration"]["ok"] is True


def test_doctor_text_reports_healthy_status() -> None:
    result = _run_cli("doctor")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "status: ok"
    assert "python: ok" in result.stdout
    assert "configuration: ok" in result.stdout


def test_doctor_returns_one_for_invalid_configuration(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("server:\n  port: 0\n", encoding="utf-8")

    result = _run_cli("doctor", "--json", "--config", str(config))

    assert result.returncode == 1
    diagnostic = json.loads(result.stdout)
    assert diagnostic["status"] == "error"
    assert diagnostic["checks"]["configuration"]["ok"] is False


def test_doctor_python_check_rejects_versions_below_floor() -> None:
    from comfyng.cli.main import _python_check

    assert _python_check((3, 13, 9)) == {
        "ok": False,
        "required": ">=3.14",
        "version": "3.13.9",
    }


@pytest.mark.parametrize("section", ("database", "storage"))
@pytest.mark.parametrize(
    ("label", "yaml_value"),
    NON_MAPPING_YAML_VALUES,
    ids=lambda value: value,
)
def test_doctor_structures_non_mapping_section_errors_without_traceback(
    tmp_path: Path,
    section: str,
    label: str,
    yaml_value: str,
) -> None:
    config = tmp_path / f"{section}-{label}.yaml"
    config.write_text(f"{section}: {yaml_value}\n", encoding="utf-8")

    result = _run_cli("doctor", "--json", "--config", str(config))

    assert result.returncode == 1
    assert result.stderr == ""
    configuration = json.loads(result.stdout)["checks"]["configuration"]
    assert configuration["ok"] is False
    assert configuration["error"]["code"] == "CONFIGURATION_INVALID"
    assert configuration["error"]["details"][0]["type"] == "config_section_type"
    assert configuration["error"]["details"][0]["ctx"] == {"section": section}


@pytest.mark.parametrize(
    ("section", "field", "outside_name"),
    (
        ("database", "path", "outside.db"),
        ("storage", "root", "outside-storage"),
    ),
)
def test_doctor_json_structures_out_of_root_path_errors_without_traceback(
    tmp_path: Path,
    section: str,
    field: str,
    outside_name: str,
) -> None:
    data_root = tmp_path / "data"
    config = tmp_path / f"outside-{section}.yaml"
    config.write_text(
        "\n".join(
            (
                f"data_root: {json.dumps(str(data_root))}",
                f"{section}:",
                f"  {field}: {json.dumps(str(tmp_path / outside_name))}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_cli("doctor", "--json", "--config", str(config))

    assert result.returncode == 1
    assert result.stderr == ""
    diagnostic = json.loads(result.stdout)
    assert diagnostic["status"] == "error"
    configuration = diagnostic["checks"]["configuration"]
    assert configuration["ok"] is False
    assert configuration["error"]["code"] == "CONFIGURATION_INVALID"
    assert configuration["error"]["details"][0]["type"] == "value_error"
    assert configuration["error"]["details"][0]["ctx"] == {
        "error": f"{section}.{field} must remain under data_root"
    }
    assert "Traceback" not in result.stdout


def test_doctor_text_preserves_out_of_root_configuration_error(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    config = tmp_path / "outside-database.yaml"
    config.write_text(
        "\n".join(
            (
                f"data_root: {json.dumps(str(data_root))}",
                "database:",
                f"  path: {json.dumps(str(tmp_path / 'outside.db'))}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_cli("doctor", "--config", str(config))

    assert result.returncode == 1
    assert result.stderr == ""
    assert result.stdout.splitlines()[0] == "status: error"
    assert "configuration: error (Configuration validation failed.)" in result.stdout
    assert "Traceback" not in result.stdout


def test_readme_documents_data_root_precedence() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    markers = (
        "$COMFYNG_DATA_ROOT",
        "$COMFYNG_HOME",
        "an explicit `data_root` in YAML",
        "$XDG_DATA_HOME/comfyui-ng",
        "~/.local/share/comfyui-ng",
    )

    positions = [readme.index(marker) for marker in markers]
    assert positions == sorted(positions)
