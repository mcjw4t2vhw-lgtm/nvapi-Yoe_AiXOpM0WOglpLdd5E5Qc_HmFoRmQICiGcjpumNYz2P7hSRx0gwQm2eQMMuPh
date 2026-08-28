import pytest
from pydantic import ValidationError

from nvapi.models import (
    ClockInfo,
    GpuInfo,
    GpuStatus,
    MemoryInfo,
    Utilization,
    classify_status,
)


def _gpu(**overrides: object) -> GpuInfo:
    data: dict[str, object] = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "uuid": "GPU-1",
        "pci_bus_id": "00000000:01:00.0",
        "temperature_c": 40,
        "memory": MemoryInfo(total_mib=1000, used_mib=100, free_mib=900),
        "utilization": Utilization(gpu_percent=10, memory_percent=10),
        "clocks": ClockInfo(graphics_mhz=200, memory_mhz=400),
    }
    data.update(overrides)
    return GpuInfo.model_validate(data)


@pytest.mark.parametrize(
    ("temp", "mem", "expected"),
    [
        (40, 10, GpuStatus.HEALTHY),
        (79, 89.9, GpuStatus.HEALTHY),
        (80, 10, GpuStatus.DEGRADED),
        (40, 90, GpuStatus.DEGRADED),
        (90, 10, GpuStatus.CRITICAL),
        (40, 98, GpuStatus.CRITICAL),
        (95, 99, GpuStatus.CRITICAL),
    ],
)
def test_classify_status(temp: int, mem: float, expected: GpuStatus) -> None:
    assert classify_status(temp, mem) == expected


def test_memory_used_percent_and_zero_total() -> None:
    assert MemoryInfo(total_mib=200, used_mib=50, free_mib=150).used_percent == 25.0
    assert MemoryInfo(total_mib=0, used_mib=0, free_mib=0).used_percent == 0.0


def test_memory_rejects_overflow() -> None:
    with pytest.raises(ValidationError):
        MemoryInfo(total_mib=100, used_mib=80, free_mib=30)


def test_utilization_bounds() -> None:
    with pytest.raises(ValidationError):
        Utilization(gpu_percent=101, memory_percent=0)
    with pytest.raises(ValidationError):
        Utilization(gpu_percent=0, memory_percent=-1)


def test_gpu_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        _gpu(name="   ")


def test_gpu_status_follows_temperature() -> None:
    assert _gpu(temperature_c=42).status == GpuStatus.HEALTHY
    assert _gpu(temperature_c=82).status == GpuStatus.DEGRADED
    assert _gpu(temperature_c=91).status == GpuStatus.CRITICAL
