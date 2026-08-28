import pytest

from nvapi.backends.mock import MockBackend
from nvapi.errors import GpuNotFoundError
from nvapi.models import (
    ClockInfo,
    GpuInfo,
    GpuStatus,
    MemoryInfo,
    Utilization,
)
from nvapi.service import NvApi


def test_mock_inventory_and_get() -> None:
    api = NvApi.mock()
    gpus = api.list_gpus()
    assert len(gpus) == 2
    assert api.get_gpu(0).name.endswith("4090")
    assert api.get_gpu(1).status == GpuStatus.DEGRADED


def test_get_gpu_missing() -> None:
    api = NvApi.mock()
    with pytest.raises(GpuNotFoundError) as exc:
        api.get_gpu(9)
    assert exc.value.index == 9
    assert exc.value.count == 2


def test_summary_aggregates() -> None:
    api = NvApi.mock()
    summary = api.summary()
    assert summary.gpu_count == 2
    assert summary.total_memory_mib == 24564 + 10240
    assert summary.used_memory_mib == 1024 + 9216
    assert summary.hottest_c == 81
    assert summary.worst_status == GpuStatus.DEGRADED
    assert summary.backend == "mock"
    assert api.driver().version == "550.54.14"


def test_empty_inventory_summary() -> None:
    api = NvApi(MockBackend(gpus=[]))
    summary = api.summary()
    assert summary.gpu_count == 0
    assert summary.hottest_c is None
    assert summary.worst_status is None


def test_summary_picks_critical_as_worst() -> None:
    hot = GpuInfo(
        index=0,
        name="Hot GPU",
        uuid="GPU-hot",
        pci_bus_id="00000000:01:00.0",
        temperature_c=96,
        memory=MemoryInfo(total_mib=100, used_mib=10, free_mib=90),
        utilization=Utilization(gpu_percent=1, memory_percent=1),
        clocks=ClockInfo(graphics_mhz=1, memory_mhz=1),
    )
    api = NvApi(MockBackend(gpus=[hot]))
    assert api.summary().worst_status == GpuStatus.CRITICAL


def test_auto_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nvapi.backends.nvidia_smi.shutil.which", lambda _name: None)
    api = NvApi.auto(allow_mock=True)
    assert api.backend.name == "mock"


def test_auto_without_mock_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nvapi.backends.nvidia_smi.shutil.which", lambda _name: None)
    with pytest.raises(Exception):
        NvApi.auto(allow_mock=False)
