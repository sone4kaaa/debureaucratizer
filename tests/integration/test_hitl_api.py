"""Интеграционные тесты HITL-роутов: очередь, confirm/reject, корректировки."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from backend.api.deps import get_db
from backend.api.main import app
from backend.db.models import User

pytestmark = pytest.mark.integration

EDITOR = {"email": "editor@example.com", "username": "editor1", "password": "Secret123"}
PLAIN = {"email": "plain2@example.com", "username": "plain2", "password": "Secret123"}


@pytest.fixture()
def client(test_db):
    async def override_get_db():
        async with test_db() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def editor_token(client: TestClient, test_db) -> str:
    resp = client.post("/auth/register", json=EDITOR)
    token = resp.json()["access_token"]

    async def _do() -> None:
        async with test_db() as session:
            await session.execute(
                update(User).where(User.email == EDITOR["email"]).values(role="editor")
            )
            await session.commit()

    asyncio.run(_do())
    return token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _push_pending(test_db, text: str = "Канцелярский текст") -> str:
    async def _do() -> str:
        from backend.hitl.active_learning import push_uncertain

        async with test_db() as session:
            example = await push_uncertain(
                session,
                {
                    "source_lang": "bureau",
                    "target_lang": "plain",
                    "source_text": text,
                    "translated_text": "Машинный вариант",
                    "critic_score": 4.0,
                    "validation_passed": False,
                },
            )
            return example.id

    return asyncio.run(_do())


def test_hitl_forbidden_for_plain_user(client: TestClient) -> None:
    resp = client.post("/auth/register", json=PLAIN)
    token = resp.json()["access_token"]
    assert client.get("/hitl/uncertain", headers=_auth(token)).status_code == 403
    assert (
        client.post(
            "/hitl/save-correction",
            json={"source_text": "a", "corrected_text": "b"},
            headers=_auth(token),
        ).status_code
        == 403
    )


def test_hitl_queue_confirm_reject_stats(
    client: TestClient, editor_token: str, test_db
) -> None:
    first = _push_pending(test_db, "Текст один")
    second = _push_pending(test_db, "Текст два")

    resp = client.get("/hitl/uncertain", headers=_auth(editor_token))
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    assert items[0]["critic_score"] == 4.0

    resp = client.post(
        f"/hitl/uncertain/{first}/confirm",
        json={"corrected_translation": "Человеческая правка"},
        headers=_auth(editor_token),
    )
    assert resp.json() == {"id": first, "status": "confirmed"}

    resp = client.post(f"/hitl/uncertain/{second}/reject", headers=_auth(editor_token))
    assert resp.json() == {"id": second, "status": "rejected"}

    resp = client.get("/hitl/uncertain/stats", headers=_auth(editor_token))
    assert resp.json() == {"pending": 0, "confirmed": 1, "rejected": 1}

    # confirm пополнил TM человеческой правкой
    async def _check_tm() -> str:
        from backend.data.translation_memory import tm_store

        async with test_db() as session:
            rows = await tm_store.get_all(session, "bureau", "plain")
            return rows[0].target_text

    assert asyncio.run(_check_tm()) == "Человеческая правка"


def test_hitl_confirm_unknown_404(client: TestClient, editor_token: str) -> None:
    resp = client.post(
        "/hitl/uncertain/нет-такого/confirm", json={}, headers=_auth(editor_token)
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "uncertain_example_not_found"


def test_hitl_save_correction(client: TestClient, editor_token: str) -> None:
    resp = client.post(
        "/hitl/save-correction",
        json={"source_text": "Осуществить оплату", "corrected_text": "Оплатить"},
        headers=_auth(editor_token),
    )
    assert resp.status_code == 200
    assert resp.json() == {"added": 1}
