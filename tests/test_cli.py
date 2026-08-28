import json

from nvapi.cli import run


def test_cli_list_mock(capsys) -> None:
    assert run(["--backend", "mock", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2


def test_cli_get_and_summary(capsys) -> None:
    assert run(["--backend", "mock", "get", "1"]) == 0
    gpu = json.loads(capsys.readouterr().out)
    assert gpu["index"] == 1

    assert run(["--backend", "mock", "summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["gpu_count"] == 2


def test_cli_driver(capsys) -> None:
    assert run(["--backend", "mock", "driver"]) == 0
    driver = json.loads(capsys.readouterr().out)
    assert driver["version"] == "550.54.14"


def test_cli_missing_gpu(capsys) -> None:
    assert run(["--backend", "mock", "get", "5"]) == 1
    err = capsys.readouterr().err
    assert "GPU 5 not found" in err
