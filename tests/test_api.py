from fastapi.testclient import TestClient

from nvapi.api import create_app
from nvapi.backends.mock import MockBackend
from nvapi.service import NvApi


def _client(api: NvApi | None = None) -> TestClient:
    return TestClient(create_app(api or NvApi.mock()))


def test_health_and_list() -> None:
    client = _client()
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["summary"]["gpu_count"] == 2

    listing = client.get("/gpus")
    assert listing.status_code == 200
    assert len(listing.json()) == 2


def test_get_gpu_and_driver() -> None:
    client = _client()
    gpu = client.get("/gpus/0")
    assert gpu.status_code == 200
    assert gpu.json()["index"] == 0
    driver = client.get("/driver")
    assert driver.json()["backend"] == "mock"


def test_missing_gpu_is_404() -> None:
    client = _client()
    response = client.get("/gpus/99")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_negative_index_is_422() -> None:
    client = _client()
    response = client.get("/gpus/-1")
    assert response.status_code == 422


def test_empty_inventory_health() -> None:
    client = _client(NvApi(MockBackend(gpus=[])))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["summary"]["gpu_count"] == 0
