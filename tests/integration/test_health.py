import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_languages(client: TestClient) -> None:
    resp = client.get("/languages")
    assert resp.status_code == 200
    body = resp.json()
    assert "bureau" in body["styles"]
    assert "plain" in body["styles"]
