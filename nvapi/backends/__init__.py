"""GPU data backends."""

from nvapi.backends.base import GpuBackend
from nvapi.backends.mock import MockBackend
from nvapi.backends.nvidia_smi import NvidiaSmiBackend

__all__ = ["GpuBackend", "MockBackend", "NvidiaSmiBackend"]
