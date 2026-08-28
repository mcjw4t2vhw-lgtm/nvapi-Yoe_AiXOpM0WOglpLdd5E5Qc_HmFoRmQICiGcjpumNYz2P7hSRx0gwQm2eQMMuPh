"""Domain models for GPU inventory."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class GpuStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class MemoryInfo(BaseModel):
    total_mib: int = Field(ge=0)
    used_mib: int = Field(ge=0)
    free_mib: int = Field(ge=0)

    @model_validator(mode="after")
    def used_and_free_fit_total(self) -> MemoryInfo:
        if self.used_mib + self.free_mib > self.total_mib:
            raise ValueError("used_mib + free_mib cannot exceed total_mib")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def used_percent(self) -> float:
        if self.total_mib == 0:
            return 0.0
        return round(100.0 * self.used_mib / self.total_mib, 2)


class Utilization(BaseModel):
    gpu_percent: int = Field(ge=0, le=100)
    memory_percent: int = Field(ge=0, le=100)


class ClockInfo(BaseModel):
    graphics_mhz: int = Field(ge=0)
    memory_mhz: int = Field(ge=0)


class DriverInfo(BaseModel):
    version: str
    backend: str

    @field_validator("version")
    @classmethod
    def version_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("driver version cannot be blank")
        return stripped


class GpuInfo(BaseModel):
    index: int = Field(ge=0)
    name: str
    uuid: str
    pci_bus_id: str
    temperature_c: int = Field(ge=0, le=120)
    memory: MemoryInfo
    utilization: Utilization
    clocks: ClockInfo

    @field_validator("name", "uuid", "pci_bus_id")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be blank")
        return stripped

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> GpuStatus:
        return classify_status(self.temperature_c, self.memory.used_percent)


class InventorySummary(BaseModel):
    gpu_count: int = Field(ge=0)
    total_memory_mib: int = Field(ge=0)
    used_memory_mib: int = Field(ge=0)
    hottest_c: int | None = None
    worst_status: GpuStatus | None = None
    backend: str


def classify_status(temperature_c: int, memory_used_percent: float) -> GpuStatus:
    """Map temperature and VRAM pressure onto a traffic-light status."""
    if temperature_c >= 90 or memory_used_percent >= 98:
        return GpuStatus.CRITICAL
    if temperature_c >= 80 or memory_used_percent >= 90:
        return GpuStatus.DEGRADED
    return GpuStatus.HEALTHY
