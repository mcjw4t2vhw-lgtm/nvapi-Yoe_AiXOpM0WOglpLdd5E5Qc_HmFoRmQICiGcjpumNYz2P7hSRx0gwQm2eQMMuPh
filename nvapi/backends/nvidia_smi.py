"""Read GPU inventory from `nvidia-smi` CSV output."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

from nvapi.errors import BackendUnavailableError, NvApiError
from nvapi.models import ClockInfo, DriverInfo, GpuInfo, MemoryInfo, Utilization

QUERY_FIELDS = [
    "name",
    "uuid",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "clocks.gr",
    "clocks.mem",
    "driver_version",
    "pci.bus_id",
]

NVIDIA_SMI_ARGS = [
    "--query-gpu=" + ",".join(QUERY_FIELDS),
    "--format=csv,noheader,nounits",
]


def _default_runner(binary: str, extra: list[str], timeout: float) -> str:
    completed = subprocess.run(
        [binary, *extra],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown error").strip()
        raise NvApiError(f"nvidia-smi failed: {detail}")
    return completed.stdout


def parse_int(raw: str, field: str) -> int:
    value = raw.strip()
    if value in {"", "N/A", "[N/A]", "na", "NA"}:
        return 0
    try:
        return int(float(value))
    except ValueError as exc:
        raise NvApiError(f"invalid integer for {field}: {raw!r}") from exc


def parse_csv_row(row: str, index: int) -> tuple[GpuInfo, str]:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != len(QUERY_FIELDS):
        raise NvApiError(
            f"expected {len(QUERY_FIELDS)} CSV columns, got {len(parts)} in row {index}: {row!r}"
        )
    (
        name,
        uuid,
        mem_total,
        mem_used,
        mem_free,
        util_gpu,
        util_mem,
        temp,
        clock_gr,
        clock_mem,
        driver,
        pci,
    ) = parts
    gpu = GpuInfo(
        index=index,
        name=name,
        uuid=uuid,
        pci_bus_id=pci,
        temperature_c=parse_int(temp, "temperature.gpu"),
        memory=MemoryInfo(
            total_mib=parse_int(mem_total, "memory.total"),
            used_mib=parse_int(mem_used, "memory.used"),
            free_mib=parse_int(mem_free, "memory.free"),
        ),
        utilization=Utilization(
            gpu_percent=min(parse_int(util_gpu, "utilization.gpu"), 100),
            memory_percent=min(parse_int(util_mem, "utilization.memory"), 100),
        ),
        clocks=ClockInfo(
            graphics_mhz=parse_int(clock_gr, "clocks.gr"),
            memory_mhz=parse_int(clock_mem, "clocks.mem"),
        ),
    )
    return gpu, driver


def parse_csv(output: str) -> tuple[list[GpuInfo], str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise NvApiError("nvidia-smi returned no GPU rows")
    gpus: list[GpuInfo] = []
    driver = ""
    for index, line in enumerate(lines):
        gpu, row_driver = parse_csv_row(line, index)
        gpus.append(gpu)
        driver = row_driver or driver
    if not driver:
        raise NvApiError("nvidia-smi did not report a driver version")
    return gpus, driver


class NvidiaSmiBackend:
    name = "nvidia-smi"

    def __init__(
        self,
        binary: str = "nvidia-smi",
        runner: Callable[[str, list[str], float], str] | None = None,
        which: Callable[[str], str | None] | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.binary = binary
        self._runner = runner or _default_runner
        self._which = which or shutil.which
        self.timeout = timeout
        self._cache: tuple[list[GpuInfo], str] | None = None

    def _load(self) -> tuple[list[GpuInfo], str]:
        if self._cache is not None:
            return self._cache
        if self._which(self.binary) is None:
            raise BackendUnavailableError(f"{self.binary} is not installed")
        output = self._runner(self.binary, NVIDIA_SMI_ARGS, self.timeout)
        self._cache = parse_csv(output)
        return self._cache

    def list_gpus(self) -> list[GpuInfo]:
        gpus, _driver = self._load()
        return list(gpus)

    def driver(self) -> DriverInfo:
        _gpus, version = self._load()
        return DriverInfo(version=version, backend=self.name)
