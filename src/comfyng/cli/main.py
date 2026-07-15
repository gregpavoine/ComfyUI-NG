from __future__ import annotations

import typer


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


@app.command()
def doctor() -> None:
    """Check the local installation and runtime prerequisites."""


@app.command()
def benchmark() -> None:
    """Run reproducible local benchmarks."""


@models_app.command("list")
def list_models() -> None:
    """List registered models."""


@models_app.command("inspect")
def inspect_model() -> None:
    """Inspect model metadata and capabilities."""


@models_app.command("import")
def import_model() -> None:
    """Import a local model asset."""


@models_app.command("download")
def download_model() -> None:
    """Download a model through an enabled provider."""


@plugins_app.command("list")
def list_plugins() -> None:
    """List discovered plugins."""


@plugins_app.command("install")
def install_plugin() -> None:
    """Install a plugin package."""


@plugins_app.command("disable")
def disable_plugin() -> None:
    """Disable an installed plugin."""


@jobs_app.command("list")
def list_jobs() -> None:
    """List known jobs."""


@jobs_app.command("cancel")
def cancel_job() -> None:
    """Cancel a queued or running job."""


@cache_app.command("inspect")
def inspect_cache() -> None:
    """Inspect cache usage."""


@cache_app.command("clean")
def clean_cache() -> None:
    """Clean eligible cache entries."""


@workers_app.command("status")
def worker_status() -> None:
    """Show worker status."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
