import pytest
from fastapi.testclient import TestClient

from backend.api.deps import get_db
from backend.api.main import app

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(test_db):
    async def override_get_db():
        async with test_db() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


USER = {"email": "user@example.com", "username": "user1", "password": "Secret123"}


def test_register_login_me_flow(client: TestClient) -> None:
    resp = client.post("/auth/register", json=USER)
    assert resp.status_code == 201
    token = resp.json()["access_token"]

    resp = client.post("/auth/login", json={"email": USER["email"], "password": USER["password"]})
    assert resp.status_code == 200
    token = resp.json()["access_token"]

    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == USER["email"]
    assert body["role"] == "user"


def test_register_duplicate_409(client: TestClient) -> None:
    client.post("/auth/register", json=USER)
    resp = client.post("/auth/register", json=USER)
    assert resp.status_code == 409


def test_register_weak_password_422(client: TestClient) -> None:
    resp = client.post(
        "/auth/register",
        json={"email": "w@example.com", "username": "weak", "password": "short"},
    )
    assert resp.status_code == 422


def test_login_wrong_password_401(client: TestClient) -> None:
    client.post("/auth/register", json=USER)
    resp = client.post(
        "/auth/login", json={"email": USER["email"], "password": "Wrong1234"}
    )
    assert resp.status_code == 401


def test_me_without_token_401(client: TestClient) -> None:
    resp = client.get("/auth/me")
    assert resp.status_code == 401
