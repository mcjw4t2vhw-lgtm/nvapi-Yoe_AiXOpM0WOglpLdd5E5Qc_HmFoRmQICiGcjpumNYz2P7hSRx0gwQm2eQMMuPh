"""High-level GPU inventory service."""

from __future__ import annotations

from nvapi.backends.base import GpuBackend
from nvapi.backends.mock import MockBackend
from nvapi.backends.nvidia_smi import NvidiaSmiBackend
from nvapi.errors import BackendUnavailableError, GpuNotFoundError
from nvapi.models import DriverInfo, GpuInfo, GpuStatus, InventorySummary


STATUS_RANK = {
    GpuStatus.HEALTHY: 0,
    GpuStatus.DEGRADED: 1,
    GpuStatus.CRITICAL: 2,
}


class NvApi:
    def __init__(self, backend: GpuBackend) -> None:
        self.backend = backend

    @classmethod
    def mock(cls, **kwargs: object) -> NvApi:
        return cls(MockBackend(**kwargs))  # type: ignore[arg-type]

    @classmethod
    def nvidia_smi(cls, **kwargs: object) -> NvApi:
        return cls(NvidiaSmiBackend(**kwargs))  # type: ignore[arg-type]

    @classmethod
    def auto(cls, allow_mock: bool = True) -> NvApi:
        try:
            api = cls.nvidia_smi()
            api.list_gpus()
            return api
        except (BackendUnavailableError, OSError):
            if not allow_mock:
                raise
            return cls.mock()

    def list_gpus(self) -> list[GpuInfo]:
        return self.backend.list_gpus()

    def get_gpu(self, index: int) -> GpuInfo:
        gpus = self.list_gpus()
        for gpu in gpus:
            if gpu.index == index:
                return gpu
        raise GpuNotFoundError(index, len(gpus))

    def driver(self) -> DriverInfo:
        return self.backend.driver()

    def summary(self) -> InventorySummary:
        gpus = self.list_gpus()
        if not gpus:
            return InventorySummary(
                gpu_count=0,
                total_memory_mib=0,
                used_memory_mib=0,
                hottest_c=None,
                worst_status=None,
                backend=self.backend.name,
            )
        worst = max(gpus, key=lambda gpu: STATUS_RANK[gpu.status])
        hottest = max(gpu.temperature_c for gpu in gpus)
        return InventorySummary(
            gpu_count=len(gpus),
            total_memory_mib=sum(gpu.memory.total_mib for gpu in gpus),
            used_memory_mib=sum(gpu.memory.used_mib for gpu in gpus),
            hottest_c=hottest,
            worst_status=worst.status,
            backend=self.backend.name,
        )
