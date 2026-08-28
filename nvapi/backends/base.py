"""Backend protocol."""

from typing import Protocol

from nvapi.models import DriverInfo, GpuInfo


class GpuBackend(Protocol):
    name: str

    def list_gpus(self) -> list[GpuInfo]:
        """Return every visible GPU."""

    def driver(self) -> DriverInfo:
        """Return driver metadata for this backend."""
