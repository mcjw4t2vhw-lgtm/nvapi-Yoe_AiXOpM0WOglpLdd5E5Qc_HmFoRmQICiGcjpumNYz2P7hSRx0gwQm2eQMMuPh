"""Deterministic in-memory GPUs for tests and GPU-less hosts."""

from __future__ import annotations

from copy import deepcopy

from nvapi.models import ClockInfo, DriverInfo, GpuInfo, MemoryInfo, Utilization

DEFAULT_GPUS = [
    GpuInfo(
        index=0,
        name="NVIDIA GeForce RTX 4090",
        uuid="GPU-4090-mock-0",
        pci_bus_id="00000000:01:00.0",
        temperature_c=42,
        memory=MemoryInfo(total_mib=24564, used_mib=1024, free_mib=23540),
        utilization=Utilization(gpu_percent=12, memory_percent=5),
        clocks=ClockInfo(graphics_mhz=210, memory_mhz=405),
    ),
    GpuInfo(
        index=1,
        name="NVIDIA GeForce RTX 3080",
        uuid="GPU-3080-mock-1",
        pci_bus_id="00000000:02:00.0",
        temperature_c=81,
        memory=MemoryInfo(total_mib=10240, used_mib=9216, free_mib=1024),
        utilization=Utilization(gpu_percent=88, memory_percent=91),
        clocks=ClockInfo(graphics_mhz=1710, memory_mhz=9501),
    ),
]


class MockBackend:
    name = "mock"

    def __init__(self, gpus: list[GpuInfo] | None = None, driver_version: str = "550.54.14") -> None:
        source = DEFAULT_GPUS if gpus is None else gpus
        self._gpus = deepcopy(source)
        self._driver_version = driver_version

    def list_gpus(self) -> list[GpuInfo]:
        return deepcopy(self._gpus)

    def driver(self) -> DriverInfo:
        return DriverInfo(version=self._driver_version, backend=self.name)
