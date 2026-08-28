"""Public exception types."""


class NvApiError(Exception):
    """Base error for all nvapi failures."""


class GpuNotFoundError(NvApiError):
    """Raised when a GPU index is out of range."""

    def __init__(self, index: int, count: int) -> None:
        self.index = index
        self.count = count
        super().__init__(f"GPU {index} not found (inventory has {count} device(s))")


class BackendUnavailableError(NvApiError):
    """Raised when the selected backend cannot run on this host."""
