from __future__ import annotations

import importlib
import subprocess
from types import SimpleNamespace

from comfyng.resources.hardware import (
    CommandResult,
    HardwareInventory,
    parse_nvidia_smi_csv,
    probe_hardware,
)


GIB = 1024**3


def test_probe_current_non_nvidia_host_without_torch_import(monkeypatch) -> None:
    imported: list[str] = []
    real_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        imported.append(name)
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("hardware probe must not import Torch")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    inventory = probe_hardware(
        command_runner=lambda _command, _timeout: CommandResult(
            returncode=127,
            stdout="",
            stderr="nvidia-smi missing",
        )
    )

    assert isinstance(inventory, HardwareInventory)
    assert inventory.cpu.physical_cores >= 1
    assert inventory.cpu.logical_cores >= inventory.cpu.physical_cores
    assert inventory.cpu.architecture
    assert inventory.memory.total_bytes > 1024**2
    assert 0 <= inventory.memory.available_bytes <= inventory.memory.total_bytes
    assert inventory.gpus == ()
    assert all(not name.startswith("torch") for name in imported)


def test_nvidia_fixture_parses_supported_and_unsupported_fields() -> None:
    csv = "\n".join(
        (
            "0, GPU-aaa, NVIDIA RTX 3080 Ti, 12288, 11000, 17, 63, 244.50, 8.6, 555.42",
            "1, GPU-bbb, Fixture GPU, 24576, 23000, [N/A], N/A, N/A, [Not Supported], 555.42",
        )
    )

    gpus = parse_nvidia_smi_csv(csv)

    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].uuid == "GPU-aaa"
    assert gpus[0].total_vram_mb == 12_288
    assert gpus[0].free_vram_mb == 11_000
    assert gpus[0].utilization_percent == 17.0
    assert gpus[0].temperature_c == 63.0
    assert gpus[0].power_watts == 244.5
    assert gpus[0].compute_capability == "8.6"
    assert gpus[0].driver_version == "555.42"

    assert gpus[1].utilization_percent is None
    assert gpus[1].temperature_c is None
    assert gpus[1].power_watts is None
    assert gpus[1].compute_capability is None


def test_probe_uses_fixture_nvidia_inventory() -> None:
    csv = "0, GPU-a, Fixture GPU, 12288, 10000, 20, 55, 180, 8.6, 555.42\n"
    calls: list[tuple[tuple[str, ...], float]] = []

    def runner(command: tuple[str, ...], timeout: float) -> CommandResult:
        calls.append((command, timeout))
        return CommandResult(returncode=0, stdout=csv, stderr="")

    inventory = probe_hardware(command_runner=runner)

    assert len(inventory.gpus) == 1
    assert inventory.gpus[0].name == "Fixture GPU"
    assert calls
    assert calls[0][0][0] == "nvidia-smi"
    assert "--format=csv,noheader,nounits" in calls[0][0]
    assert 0 < calls[0][1] <= 5


def test_probe_prefers_psutil_when_available(monkeypatch) -> None:
    fake_psutil = SimpleNamespace(
        cpu_count=lambda *, logical: 12 if logical else 6,
        cpu_percent=lambda *, interval: 23.5,
        virtual_memory=lambda: SimpleNamespace(total=32 * GIB, available=20 * GIB),
        swap_memory=lambda: SimpleNamespace(total=4 * GIB, free=3 * GIB),
    )
    real_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "psutil":
            return fake_psutil
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    inventory = probe_hardware(
        command_runner=lambda _command, _timeout: CommandResult(127, "", "missing")
    )

    assert inventory.cpu.physical_cores == 6
    assert inventory.cpu.logical_cores == 12
    assert inventory.cpu.load_percent == 23.5
    assert inventory.memory.total_bytes == 32 * GIB
    assert inventory.memory.available_bytes == 20 * GIB
    assert inventory.memory.swap_total_bytes == 4 * GIB
    assert inventory.memory.swap_free_bytes == 3 * GIB


def test_probe_retries_with_basic_query_when_optional_field_is_unsupported() -> None:
    basic_csv = "0, GPU-a, Older GPU, 8192, 7000, 10, 52, 120, 470.00\n"
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], _timeout: float) -> CommandResult:
        calls.append(command)
        if "compute_cap" in command[1]:
            return CommandResult(1, "", "Field 'compute_cap' is not a valid field")
        return CommandResult(0, basic_csv, "")

    inventory = probe_hardware(command_runner=runner)

    assert len(calls) == 2
    assert len(inventory.gpus) == 1
    assert inventory.gpus[0].compute_capability is None
    assert inventory.gpus[0].driver_version == "470.00"


def test_malformed_or_failed_nvidia_probe_is_non_fatal() -> None:
    failed = probe_hardware(
        command_runner=lambda _command, _timeout: CommandResult(
            returncode=1,
            stdout="",
            stderr="driver unavailable",
        )
    )
    malformed = probe_hardware(
        command_runner=lambda _command, _timeout: CommandResult(
            returncode=0,
            stdout="this is not a supported csv row",
            stderr="",
        )
    )

    assert failed.gpus == ()
    assert malformed.gpus == ()


def test_default_runner_has_a_bounded_timeout(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe_hardware()

    assert seen["timeout"] == 2.0
    assert seen["check"] is False
    assert seen["text"] is True


def test_inventory_contract_rejects_impossible_values() -> None:
    from comfyng.resources.hardware import CpuInventory, GpuDevice, MemoryInventory

    try:
        CpuInventory(physical_cores=8, logical_cores=4, architecture="x86_64")
    except ValueError:
        pass
    else:
        raise AssertionError("logical core count below physical must be rejected")

    try:
        MemoryInventory(
            total_bytes=4 * GIB,
            available_bytes=5 * GIB,
            swap_total_bytes=0,
            swap_free_bytes=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("available RAM above total must be rejected")

    memory = MemoryInventory(
        total_bytes=8 * GIB,
        available_bytes=4 * GIB,
        swap_total_bytes=0,
        swap_free_bytes=0,
    )
    gpu0 = GpuDevice(0, "GPU 0", 1024, 1024)
    gpu1 = GpuDevice(1, "GPU 1", 1024, 1024)
    try:
        HardwareInventory(
            cpu=CpuInventory(4, 8, "x86_64"),
            memory=memory,
            gpus=(gpu1, gpu0),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("GPU indexes must have deterministic ordering")
