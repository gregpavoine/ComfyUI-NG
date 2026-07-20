from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib
import io
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Protocol


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_finite_number(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or None")
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite non-negative number or None")


@dataclass(frozen=True, slots=True)
class CpuInventory:
    physical_cores: int
    logical_cores: int
    architecture: str
    processor: str | None = None
    simd_instructions: tuple[str, ...] = ()
    numa_nodes: int | None = None
    load_percent: float | None = None

    def __post_init__(self) -> None:
        _require_int("physical_cores", self.physical_cores, minimum=1)
        _require_int("logical_cores", self.logical_cores, minimum=1)
        if self.logical_cores < self.physical_cores:
            raise ValueError("logical_cores must be >= physical_cores")
        if not isinstance(self.architecture, str) or not self.architecture.strip():
            raise ValueError("architecture must be a non-empty string")
        if self.processor is not None and not isinstance(self.processor, str):
            raise ValueError("processor must be a string or None")
        if not isinstance(self.simd_instructions, tuple) or not all(
            isinstance(item, str) and item for item in self.simd_instructions
        ):
            raise ValueError("simd_instructions must be a tuple of non-empty strings")
        if self.numa_nodes is not None:
            _require_int("numa_nodes", self.numa_nodes, minimum=1)
        _optional_finite_number("load_percent", self.load_percent)

    @property
    def threads(self) -> int:
        return self.logical_cores


@dataclass(frozen=True, slots=True)
class MemoryInventory:
    total_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int

    def __post_init__(self) -> None:
        _require_int("total_bytes", self.total_bytes, minimum=1)
        _require_int("available_bytes", self.available_bytes)
        _require_int("swap_total_bytes", self.swap_total_bytes)
        _require_int("swap_free_bytes", self.swap_free_bytes)
        if self.available_bytes > self.total_bytes:
            raise ValueError("available_bytes cannot exceed total_bytes")
        if self.swap_free_bytes > self.swap_total_bytes:
            raise ValueError("swap_free_bytes cannot exceed swap_total_bytes")

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    total_vram_mb: int
    free_vram_mb: int
    uuid: str | None = None
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    backend: str = "cuda"

    def __post_init__(self) -> None:
        _require_int("index", self.index)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _require_int("total_vram_mb", self.total_vram_mb, minimum=1)
        _require_int("free_vram_mb", self.free_vram_mb)
        if self.free_vram_mb > self.total_vram_mb:
            raise ValueError("free_vram_mb cannot exceed total_vram_mb")
        for name in ("uuid", "compute_capability", "driver_version"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        _optional_finite_number("utilization_percent", self.utilization_percent)
        _optional_finite_number("temperature_c", self.temperature_c)
        _optional_finite_number("power_watts", self.power_watts)
        if self.utilization_percent is not None and self.utilization_percent > 100:
            raise ValueError("utilization_percent cannot exceed 100")

    @property
    def used_vram_mb(self) -> int:
        return self.total_vram_mb - self.free_vram_mb


@dataclass(frozen=True, slots=True)
class DiskDevice:
    path: str
    total_bytes: int | None = None
    free_bytes: int | None = None
    rotational: bool | None = None
    read_bytes_per_second: int | None = None
    write_bytes_per_second: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be a non-empty string")
        for name in (
            "total_bytes",
            "free_bytes",
            "read_bytes_per_second",
            "write_bytes_per_second",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_int(name, value)
        if (
            self.total_bytes is not None
            and self.free_bytes is not None
            and self.free_bytes > self.total_bytes
        ):
            raise ValueError("free_bytes cannot exceed total_bytes")
        if self.rotational is not None and not isinstance(self.rotational, bool):
            raise ValueError("rotational must be a boolean or None")


@dataclass(frozen=True, slots=True)
class NetworkInventory:
    bandwidth_bytes_per_second: int | None = None
    utilization_percent: float | None = None

    def __post_init__(self) -> None:
        if self.bandwidth_bytes_per_second is not None:
            _require_int("bandwidth_bytes_per_second", self.bandwidth_bytes_per_second)
        _optional_finite_number("utilization_percent", self.utilization_percent)
        if self.utilization_percent is not None and self.utilization_percent > 100:
            raise ValueError("utilization_percent cannot exceed 100")


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    cpu: CpuInventory
    memory: MemoryInventory
    gpus: tuple[GpuDevice, ...] = ()
    disks: tuple[DiskDevice, ...] = ()
    network: NetworkInventory = NetworkInventory()

    def __post_init__(self) -> None:
        if not isinstance(self.cpu, CpuInventory):
            raise ValueError("cpu must be a CpuInventory")
        if not isinstance(self.memory, MemoryInventory):
            raise ValueError("memory must be a MemoryInventory")
        if not isinstance(self.gpus, tuple) or not all(
            isinstance(item, GpuDevice) for item in self.gpus
        ):
            raise ValueError("gpus must be a tuple of GpuDevice values")
        indexes = tuple(item.index for item in self.gpus)
        if len(indexes) != len(set(indexes)):
            raise ValueError("GPU indexes must be unique")
        if indexes != tuple(sorted(indexes)):
            raise ValueError("GPU indexes must be sorted")
        if not isinstance(self.disks, tuple) or not all(
            isinstance(item, DiskDevice) for item in self.disks
        ):
            raise ValueError("disks must be a tuple of DiskDevice values")
        if not isinstance(self.network, NetworkInventory):
            raise ValueError("network must be a NetworkInventory")

    @property
    def physical_cpu_cores(self) -> int:
        return self.cpu.physical_cores

    @property
    def logical_cpu_cores(self) -> int:
        return self.cpu.logical_cores

    @property
    def total_ram_bytes(self) -> int:
        return self.memory.total_bytes

    @property
    def available_ram_bytes(self) -> int:
        return self.memory.available_bytes

    def gpu(self, index: int) -> GpuDevice | None:
        return next((item for item in self.gpus if item.index == index), None)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def __call__(self, command: tuple[str, ...], timeout: float) -> CommandResult: ...


_UNSUPPORTED = frozenset(
    {
        "",
        "-",
        "n/a",
        "[n/a]",
        "not supported",
        "[not supported]",
        "unknown",
    }
)


def _optional_text(value: str) -> str | None:
    normalized = value.strip()
    if normalized.casefold() in _UNSUPPORTED:
        return None
    return normalized


def _optional_float(value: str) -> float | None:
    normalized = _optional_text(value)
    if normalized is None:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    if parsed < 0 or parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _required_int(value: str) -> int | None:
    parsed = _optional_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _parse_nvidia_smi_csv(
    value: str,
    *,
    includes_compute_capability: bool,
) -> tuple[GpuDevice, ...]:
    devices: list[GpuDevice] = []
    reader = csv.reader(io.StringIO(value), skipinitialspace=True)
    for row in reader:
        expected_columns = 10 if includes_compute_capability else 9
        if len(row) != expected_columns:
            continue
        index = _required_int(row[0])
        uuid = _optional_text(row[1])
        name = _optional_text(row[2])
        total = _required_int(row[3])
        free = _required_int(row[4])
        if index is None or name is None or total is None or free is None:
            continue
        try:
            devices.append(
                GpuDevice(
                    index=index,
                    uuid=uuid,
                    name=name,
                    total_vram_mb=total,
                    free_vram_mb=free,
                    utilization_percent=_optional_float(row[5]),
                    temperature_c=_optional_float(row[6]),
                    power_watts=_optional_float(row[7]),
                    compute_capability=(
                        _optional_text(row[8]) if includes_compute_capability else None
                    ),
                    driver_version=_optional_text(
                        row[9] if includes_compute_capability else row[8]
                    ),
                )
            )
        except ValueError:
            continue
    return tuple(sorted(devices, key=lambda item: item.index))


def parse_nvidia_smi_csv(value: str) -> tuple[GpuDevice, ...]:
    """Parse the extended no-unit query, ignoring malformed devices safely."""

    return _parse_nvidia_smi_csv(value, includes_compute_capability=True)


_NVIDIA_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,compute_cap,driver_version",
    "--format=csv,noheader,nounits",
)
_NVIDIA_BASIC_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,driver_version",
    "--format=csv,noheader,nounits",
)


def _run_command(command: tuple[str, ...], timeout: float) -> CommandResult:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return CommandResult(returncode=127, stdout="", stderr="command unavailable")
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _probe_nvidia(command_runner: CommandRunner) -> tuple[GpuDevice, ...]:
    queries = (
        (_NVIDIA_QUERY, True),
        (_NVIDIA_BASIC_QUERY, False),
    )
    for command, includes_compute_capability in queries:
        try:
            result = command_runner(command, 2.0)
            if result.returncode != 0:
                continue
            devices = _parse_nvidia_smi_csv(
                result.stdout,
                includes_compute_capability=includes_compute_capability,
            )
        except (
            AttributeError,
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
            TypeError,
            ValueError,
        ):
            continue
        if devices:
            return devices
    return ()


def _psutil_module() -> Any | None:
    try:
        return importlib.import_module("psutil")
    except (ImportError, OSError):
        return None


def _system_output(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _darwin_memory() -> MemoryInventory | None:
    raw_total = _system_output(("/usr/sbin/sysctl", "-n", "hw.memsize"))
    if raw_total is None:
        return None
    try:
        total = int(raw_total)
    except ValueError:
        return None
    if total <= 0:
        return None

    available = 0
    vm_stat = _system_output(("/usr/bin/vm_stat",))
    if vm_stat is not None:
        page_match = re.search(r"page size of (\d+) bytes", vm_stat)
        page_size = int(page_match.group(1)) if page_match else 4096
        pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative"):
            match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.", vm_stat, re.MULTILINE)
            if match:
                pages += int(match.group(1))
        available = min(total, pages * page_size)

    swap_total = swap_free = 0
    raw_swap = _system_output(("/usr/sbin/sysctl", "-n", "vm.swapusage"))
    if raw_swap is not None:
        total_match = re.search(r"total\s*=\s*([0-9.]+)([KMG])", raw_swap)
        free_match = re.search(r"free\s*=\s*([0-9.]+)([KMG])", raw_swap)
        units = {"K": 1024, "M": 1024**2, "G": 1024**3}
        if total_match:
            swap_total = int(float(total_match.group(1)) * units[total_match.group(2)])
        if free_match:
            swap_free = int(float(free_match.group(1)) * units[free_match.group(2)])
        swap_free = min(swap_free, swap_total)
    return MemoryInventory(
        total_bytes=total,
        available_bytes=available,
        swap_total_bytes=swap_total,
        swap_free_bytes=swap_free,
    )


def _linux_memory() -> MemoryInventory | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        match = re.match(r"\s*(\d+)\s+kB\s*$", raw_value)
        if match:
            values[key] = int(match.group(1)) * 1024
    total = values.get("MemTotal", 0)
    if total <= 0:
        return None
    available = values.get("MemAvailable", values.get("MemFree", 0))
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return MemoryInventory(
        total_bytes=total,
        available_bytes=min(total, available),
        swap_total_bytes=swap_total,
        swap_free_bytes=min(swap_total, swap_free),
    )


def _fallback_memory() -> MemoryInventory:
    system = platform.system()
    if system == "Darwin" and (memory := _darwin_memory()) is not None:
        return memory
    if system == "Linux" and (memory := _linux_memory()) is not None:
        return memory
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * total_pages
    except (KeyError, OSError, TypeError, ValueError):
        total = 1
        page_size = 1
    try:
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        available = page_size * available_pages
    except (KeyError, OSError, TypeError, ValueError):
        available = 0
    return MemoryInventory(
        total_bytes=max(1, total),
        available_bytes=max(0, min(total, available)),
        swap_total_bytes=0,
        swap_free_bytes=0,
    )


def _fallback_physical_cores(logical: int) -> int:
    if platform.system() == "Darwin":
        raw = _system_output(("/usr/sbin/sysctl", "-n", "hw.physicalcpu"))
        if raw is not None:
            try:
                return max(1, min(logical, int(raw)))
            except ValueError:
                pass
    if platform.system() == "Linux":
        topology_root = Path("/sys/devices/system/cpu")
        cores: set[tuple[str, str]] = set()
        try:
            for cpu_path in topology_root.glob("cpu[0-9]*"):
                topology = cpu_path / "topology"
                package = (
                    (topology / "physical_package_id")
                    .read_text(encoding="ascii")
                    .strip()
                )
                core = (topology / "core_id").read_text(encoding="ascii").strip()
                cores.add((package, core))
        except (OSError, UnicodeError):
            cores.clear()
        if cores:
            return max(1, min(logical, len(cores)))
    return logical


def _fallback_cpu_features() -> tuple[str, ...]:
    features: set[str] = set()
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip().casefold() in {"flags", "features"}:
                    features.update(value.casefold().split())
        except (OSError, UnicodeError):
            pass
    elif platform.system() == "Darwin":
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            value = _system_output(("/usr/sbin/sysctl", "-n", key))
            if value:
                features.update(value.casefold().split())
    return tuple(sorted(features))


def _fallback_numa_nodes() -> int | None:
    if platform.system() != "Linux":
        return None
    try:
        nodes = tuple(Path("/sys/devices/system/node").glob("node[0-9]*"))
    except OSError:
        return None
    return len(nodes) or None


def _probe_cpu_and_memory() -> tuple[CpuInventory, MemoryInventory]:
    psutil = _psutil_module()
    logical = max(1, os.cpu_count() or 1)
    physical = _fallback_physical_cores(logical)
    try:
        load_percent: float | None = min(
            100.0,
            max(0.0, os.getloadavg()[0] / logical * 100),
        )
    except (AttributeError, OSError):
        load_percent = None
    memory = _fallback_memory()
    if psutil is not None:
        try:
            logical = max(1, int(psutil.cpu_count(logical=True) or logical))
            physical = max(1, int(psutil.cpu_count(logical=False) or logical))
            physical = min(physical, logical)
            measured_load = psutil.cpu_percent(interval=None)
            load_percent = float(measured_load) if measured_load is not None else None
        except (AttributeError, OSError, TypeError, ValueError):
            physical = min(physical, logical)
            load_percent = None
        try:
            virtual = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory = MemoryInventory(
                total_bytes=int(virtual.total),
                available_bytes=int(virtual.available),
                swap_total_bytes=int(swap.total),
                swap_free_bytes=int(swap.free),
            )
        except (AttributeError, OSError, TypeError, ValueError):
            memory = _fallback_memory()
    cpu = CpuInventory(
        physical_cores=physical,
        logical_cores=logical,
        architecture=platform.machine() or "unknown",
        processor=platform.processor() or None,
        simd_instructions=_fallback_cpu_features(),
        numa_nodes=_fallback_numa_nodes(),
        load_percent=load_percent,
    )
    return cpu, memory


def _probe_root_disk() -> tuple[DiskDevice, ...]:
    try:
        stat = os.statvfs(PathLikeRoot)
    except (AttributeError, OSError, ValueError):
        return ()
    total = stat.f_blocks * stat.f_frsize
    free = stat.f_bavail * stat.f_frsize
    if total <= 0:
        return ()
    return (DiskDevice(path=PathLikeRoot, total_bytes=total, free_bytes=free),)


PathLikeRoot = os.path.abspath(os.sep)


def probe_hardware(
    *,
    command_runner: CommandRunner | None = None,
) -> HardwareInventory:
    """Probe the host without initializing any ML or accelerator runtime."""

    cpu, memory = _probe_cpu_and_memory()
    runner = command_runner or _run_command
    return HardwareInventory(
        cpu=cpu,
        memory=memory,
        gpus=_probe_nvidia(runner),
        disks=_probe_root_disk(),
    )


__all__ = [
    "CommandResult",
    "CpuInventory",
    "DiskDevice",
    "GpuDevice",
    "HardwareInventory",
    "MemoryInventory",
    "NetworkInventory",
    "parse_nvidia_smi_csv",
    "probe_hardware",
]
