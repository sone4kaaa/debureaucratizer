"""Интеграционные тесты админ-роутов: glossary, TM, кэш, роли."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from backend.api.deps import get_db
from backend.api.main import app
from backend.db.models import User

pytestmark = pytest.mark.integration

ADMIN = {"email": "admin@example.com", "username": "admin1", "password": "Secret123"}
PLAIN = {"email": "plain@example.com", "username": "plain1", "password": "Secret123"}


@pytest.fixture()
def client(test_db):
    async def override_get_db():
        async with test_db() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _register(client: TestClient, user: dict) -> str:
    resp = client.post("/auth/register", json=user)
    assert resp.status_code == 201
    return resp.json()["access_token"]


def _promote(test_db, email: str, role: str) -> None:
    async def _do() -> None:
        async with test_db() as session:
            await session.execute(
                update(User).where(User.email == email).values(role=role)
            )
            await session.commit()

    asyncio.run(_do())


@pytest.fixture()
def admin_token(client: TestClient, test_db) -> str:
    token = _register(client, ADMIN)
    _promote(test_db, ADMIN["email"], "admin")
    return token


@pytest.fixture()
def user_token(client: TestClient) -> str:
    return _register(client, PLAIN)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------- роли ----------


def test_admin_routes_forbidden_for_user(client: TestClient, user_token: str) -> None:
    assert client.get("/admin/glossary/list", headers=_auth(user_token)).status_code == 403
    assert client.get("/admin/cache/stats", headers=_auth(user_token)).status_code == 403


def test_admin_routes_unauthorized(client: TestClient) -> None:
    assert client.get("/admin/glossary/list").status_code == 401


# ---------- glossary ----------


def test_glossary_crud(client: TestClient, admin_token: str) -> None:
    resp = client.post(
        "/admin/glossary/add",
        json={"source_text": "в целях", "target_text": "чтобы"},
        headers=_auth(admin_token),
    )
    assert resp.status_code == 200
    entry_id = resp.json()["id"]

    resp = client.get("/admin/glossary/list", headers=_auth(admin_token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["source_text"] == "в целях"

    resp = client.delete(f"/admin/glossary/delete/{entry_id}", headers=_auth(admin_token))
    assert resp.json() == {"deleted": True}
    resp = client.delete(f"/admin/glossary/delete/{entry_id}", headers=_auth(admin_token))
    assert resp.status_code == 404


def test_glossary_import_csv_idempotent(client: TestClient, admin_token: str) -> None:
    csv_content = (
        "source_lang,target_lang,en,tgt\n"
        "bureau,plain,осуществить,сделать\n"
        "bureau,plain,направить,отправить\n"
    )
    files = {"file": ("glossary.csv", csv_content.encode(), "text/csv")}
    resp = client.post("/admin/glossary/import-csv", files=files, headers=_auth(admin_token))
    assert resp.status_code == 200
    assert resp.json() == {"added": 2, "total": 2}
    # повторный импорт — дубликаты пропущены
    resp = client.post("/admin/glossary/import-csv", files=files, headers=_auth(admin_token))
    assert resp.json() == {"added": 0, "total": 2}


# ---------- TM ----------


def test_tm_add_list_delete(client: TestClient, admin_token: str) -> None:
    payload = {"source_text": "Настоящим уведомляем", "target_text": "Сообщаем"}
    resp = client.post("/admin/tm/add", json=payload, headers=_auth(admin_token))
    assert resp.json() == {"added": 1}
    # дубликат
    resp = client.post("/admin/tm/add", json=payload, headers=_auth(admin_token))
    assert resp.json() == {"added": 0}

    resp = client.get("/admin/tm/list", headers=_auth(admin_token))
    body = resp.json()
    assert body["total"] == 1
    entry_id = body["items"][0]["id"]

    resp = client.delete(f"/admin/tm/delete/{entry_id}", headers=_auth(admin_token))
    assert resp.json() == {"deleted": True}
    resp = client.delete(f"/admin/tm/delete/{entry_id}", headers=_auth(admin_token))
    assert resp.status_code == 404


def test_tm_import_csv_groups_by_lang(client: TestClient, admin_token: str) -> None:
    csv_content = (
        "source_lang,target_lang,en,tgt\n"
        "bureau,plain,сложное1,простое1\n"
        "legal,plain,сложное2,простое2\n"
    )
    files = {"file": ("tm.csv", csv_content.encode(), "text/csv")}
    resp = client.post("/admin/tm/import-csv", files=files, headers=_auth(admin_token))
    assert resp.json() == {"added": 2, "total": 2}


# ---------- кэш ----------


def test_cache_stats_and_clear(client: TestClient, admin_token: str, test_db) -> None:
    async def _seed() -> None:
        from backend.data.cache import cache_store

        async with test_db() as session:
            await cache_store.put(session, "bureau", "plain", "текст", "простой")

    asyncio.run(_seed())

    resp = client.get("/admin/cache/stats", headers=_auth(admin_token))
    assert resp.status_code == 200
    assert resp.json()["total_entries"] == 1

    resp = client.delete("/admin/cache/clear", headers=_auth(admin_token))
    assert resp.json() == {"deleted": 1}
