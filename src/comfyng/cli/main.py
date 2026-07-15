from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Annotated, Any, NoReturn

import typer


EX_UNAVAILABLE = 69


def _service_unavailable(service: str) -> NoReturn:
    typer.echo(
        json.dumps(
            {
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Required service is not available in this installation.",
                    "service": service,
                }
            },
            sort_keys=True,
        ),
        err=True,
    )
    raise typer.Exit(code=EX_UNAVAILABLE)


def _python_check(version_info: Sequence[int]) -> dict[str, str | bool]:
    major, minor, micro = version_info[:3]
    return {
        "ok": (major, minor) >= (3, 14),
        "required": ">=3.14",
        "version": f"{major}.{minor}.{micro}",
    }


def _configuration_check(config: Path | None) -> dict[str, Any]:
    from pydantic import ValidationError
    from yaml import YAMLError

    from comfyng.config import Settings

    try:
        settings = Settings.load(path=config)
    except ValidationError as error:
        return {
            "error": {
                "code": "CONFIGURATION_INVALID",
                "details": json.loads(
                    error.json(include_url=False, include_input=False)
                ),
                "message": "Configuration validation failed.",
            },
            "ok": False,
        }
    except (OSError, TypeError, ValueError, YAMLError) as error:
        return {
            "error": {
                "code": "CONFIGURATION_INVALID",
                "details": [],
                "message": str(error),
            },
            "ok": False,
        }
    return {
        "data_root": str(settings.data_root),
        "database": str(settings.database.path),
        "ok": True,
        "storage": str(settings.storage.root),
    }


def _render_doctor_text(diagnostic: dict[str, Any]) -> None:
    python_check = diagnostic["checks"]["python"]
    configuration_check = diagnostic["checks"]["configuration"]
    typer.echo(f"status: {diagnostic['status']}")
    typer.echo(
        "python: "
        f"{'ok' if python_check['ok'] else 'error'} "
        f"({python_check['version']}; required {python_check['required']})"
    )
    if configuration_check["ok"]:
        typer.echo(
            "configuration: ok "
            f"(data_root={configuration_check['data_root']})"
        )
    else:
        typer.echo(
            "configuration: error "
            f"({configuration_check['error']['message']})"
        )


app = typer.Typer(
    name="comfyng",
    help="ComfyUI-NG control-plane commands.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Inspect and manage model assets.", no_args_is_help=True)
plugins_app = typer.Typer(help="Inspect and manage plugins.", no_args_is_help=True)
jobs_app = typer.Typer(help="Inspect and manage jobs.", no_args_is_help=True)
cache_app = typer.Typer(help="Inspect and manage cached data.", no_args_is_help=True)
workers_app = typer.Typer(help="Inspect worker processes.", no_args_is_help=True)

app.add_typer(models_app, name="models")
app.add_typer(plugins_app, name="plugins")
app.add_typer(jobs_app, name="jobs")
app.add_typer(cache_app, name="cache")
app.add_typer(workers_app, name="workers")


@app.command()
def serve() -> None:
    """Start the local ComfyUI-NG service."""

    _service_unavailable("api")


@app.command()
def doctor(
    config: Annotated[
        Path | None,
        typer.Option("--config", dir_okay=False, help="Optional YAML configuration."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Render the diagnostic as JSON."),
    ] = False,
) -> None:
    """Check the local installation and runtime prerequisites."""

    checks = {
        "python": _python_check(sys.version_info),
        "configuration": _configuration_check(config),
    }
    healthy = all(bool(check["ok"]) for check in checks.values())
    diagnostic = {"checks": checks, "status": "ok" if healthy else "error"}
    if json_output:
        typer.echo(json.dumps(diagnostic, sort_keys=True))
    else:
        _render_doctor_text(diagnostic)
    if not healthy:
        raise typer.Exit(code=1)


@app.command()
def benchmark() -> None:
    """Run reproducible local benchmarks."""

    _service_unavailable("benchmark")


@models_app.command("list")
def list_models() -> None:
    """List registered models."""

    _service_unavailable("models")


@models_app.command("inspect")
def inspect_model() -> None:
    """Inspect model metadata and capabilities."""

    _service_unavailable("models")


@models_app.command("import")
def import_model() -> None:
    """Import a local model asset."""

    _service_unavailable("models")


@models_app.command("download")
def download_model() -> None:
    """Download a model through an enabled provider."""

    _service_unavailable("providers")


@plugins_app.command("list")
def list_plugins() -> None:
    """List discovered plugins."""

    _service_unavailable("plugins")


@plugins_app.command("install")
def install_plugin() -> None:
    """Install a plugin package."""

    _service_unavailable("plugins")


@plugins_app.command("disable")
def disable_plugin() -> None:
    """Disable an installed plugin."""

    _service_unavailable("plugins")


@jobs_app.command("list")
def list_jobs() -> None:
    """List known jobs."""

    _service_unavailable("jobs")


@jobs_app.command("cancel")
def cancel_job() -> None:
    """Cancel a queued or running job."""

    _service_unavailable("jobs")


@cache_app.command("inspect")
def inspect_cache() -> None:
    """Inspect cache usage."""

    _service_unavailable("cache")


@cache_app.command("clean")
def clean_cache() -> None:
    """Clean eligible cache entries."""

    _service_unavailable("cache")


@workers_app.command("status")
def worker_status() -> None:
    """Show worker status."""

    _service_unavailable("workers")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
