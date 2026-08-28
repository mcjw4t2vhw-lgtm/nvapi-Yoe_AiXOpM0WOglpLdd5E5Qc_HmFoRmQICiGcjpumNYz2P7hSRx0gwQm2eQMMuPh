import pytest

from nvapi.backends.nvidia_smi import (
    NvidiaSmiBackend,
    _default_runner,
    parse_csv,
    parse_int,
)
from nvapi.errors import BackendUnavailableError, NvApiError

SAMPLE = (
    "NVIDIA GeForce RTX 4090, GPU-aaa, 24564, 1024, 23540, 12, 5, 41, 210, 405, 550.54.14, 00000000:01:00.0\n"
    "NVIDIA GeForce RTX 3080, GPU-bbb, 10240, 2048, 8192, 50, 20, 55, 1710, 9501, 550.54.14, 00000000:02:00.0\n"
)


def test_parse_csv_two_gpus() -> None:
    gpus, driver = parse_csv(SAMPLE)
    assert driver == "550.54.14"
    assert len(gpus) == 2
    assert gpus[0].uuid == "GPU-aaa"
    assert gpus[1].memory.used_mib == 2048
    assert gpus[0].index == 0
    assert gpus[1].index == 1


def test_parse_csv_na_and_floats() -> None:
    row = "Name, UUID-1, 100.9, N/A, 100, [N/A], 3, 30, NA, 400, 1.0, bus"
    gpus, driver = parse_csv(row)
    assert driver == "1.0"
    assert gpus[0].memory.used_mib == 0
    assert gpus[0].utilization.gpu_percent == 0
    assert gpus[0].memory.total_mib == 100
    assert gpus[0].clocks.graphics_mhz == 0


def test_parse_csv_rejects_empty() -> None:
    with pytest.raises(NvApiError, match="no GPU rows"):
        parse_csv("\n  \n")


def test_parse_csv_rejects_wrong_column_count() -> None:
    with pytest.raises(NvApiError, match="CSV columns"):
        parse_csv("only,three,cols")


def test_parse_int_rejects_garbage() -> None:
    with pytest.raises(NvApiError, match="invalid integer"):
        parse_int("hot", "temperature.gpu")


def test_backend_uses_runner() -> None:
    backend = NvidiaSmiBackend(
        which=lambda _name: "/usr/bin/nvidia-smi",
        runner=lambda _binary, _args, _timeout: SAMPLE,
    )
    assert len(backend.list_gpus()) == 2
    assert backend.driver().backend == "nvidia-smi"
    # second call hits cache
    assert backend.list_gpus()[0].name.startswith("NVIDIA")


def test_backend_missing_binary() -> None:
    backend = NvidiaSmiBackend(which=lambda _name: None)
    with pytest.raises(BackendUnavailableError):
        backend.list_gpus()


def test_default_runner_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stderr = "no devices found\n"
        stdout = ""

    monkeypatch.setattr(
        "nvapi.backends.nvidia_smi.subprocess.run",
        lambda *_args, **_kwargs: Result(),
    )
    with pytest.raises(NvApiError, match="no devices found"):
        _default_runner("nvidia-smi", [], 1.0)


def test_runner_nonzero_exit() -> None:
    def boom(_binary: str, _args: list[str], _timeout: float) -> str:
        raise NvApiError("nvidia-smi failed: no devices")

    backend = NvidiaSmiBackend(
        which=lambda _name: "/usr/bin/nvidia-smi",
        runner=boom,
    )
    with pytest.raises(NvApiError, match="no devices"):
        backend.list_gpus()
