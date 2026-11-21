"""Интеграционные тесты API перевода: /translate, /multi, /batch, /ensemble."""

import pytest
from fastapi.testclient import TestClient

import backend.services.pipeline as pipeline
from backend.api.deps import get_db
from backend.api.main import app

pytestmark = pytest.mark.integration

USER = {"email": "tr@example.com", "username": "translator", "password": "Secret123"}


class _FakeManager:
    async def translate(
        self, prompt, source_lang, target_lang, engine=None, strict_engine=False
    ) -> dict:
        if "FAIL" in prompt:
            from backend.core.exceptions import EngineUnavailableError

            raise EngineUnavailableError("движок упал")
        return {
            "text": "Оплатите счёт до 15 марта.",
            "engine_used": "ollama",
            "model_used": "test-model",
        }


@pytest.fixture()
def client(test_db, monkeypatch):
    async def override_get_db():
        async with test_db() as session:
            yield session

    monkeypatch.setattr(pipeline, "translator_manager", _FakeManager())
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def token(client: TestClient) -> str:
    resp = client.post("/auth/register", json=USER)
    assert resp.status_code == 201
    return resp.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_translate_requires_auth(client: TestClient) -> None:
    resp = client.post("/translate", json={"text": "текст"})
    assert resp.status_code == 401


def test_translate_baseline_success(client: TestClient, token: str) -> None:
    resp = client.post(
        "/translate",
        json={"text": "Необходимо осуществить оплату в срок.", "eval_mode": "baseline"},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["translated_text"] == "Оплатите счёт до 15 марта."
    assert body["from_cache"] is False
    assert body["eval_mode"] == "baseline"
    assert body["final_success"] is True


def test_translate_cache_hit(client: TestClient, token: str) -> None:
    payload = {"text": "Уникальный текст для кэша", "eval_mode": "baseline"}
    first = client.post("/translate", json=payload, headers=_auth(token))
    assert first.status_code == 200
    second = client.post("/translate", json=payload, headers=_auth(token))
    assert second.status_code == 200
    assert second.json()["from_cache"] is True


def test_translate_unknown_mode_400(client: TestClient, token: str) -> None:
    resp = client.post(
        "/translate",
        json={"text": "текст", "eval_mode": "нет-такого"},
        headers=_auth(token),
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "unknown_eval_mode"


def test_translate_multi(client: TestClient, token: str) -> None:
    resp = client.post(
        "/translate/multi",
        json={"text": "текст", "target_langs": ["plain", "official"]},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["results"]) == {"plain", "official"}
    assert body["results"]["plain"]["translated_text"]


def test_translate_batch_with_partial_error(client: TestClient, token: str) -> None:
    resp = client.post(
        "/translate/batch",
        json={"texts": ["нормальный текст", "текст с FAIL"]},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert "translated_text" in body["results"][0]
    assert body["results"][1]["error"]["code"] == "engine_unavailable"


def test_ensemble_unknown_pair_400(client: TestClient, token: str) -> None:
    resp = client.post(
        "/translate/ensemble",
        json={"text": "текст", "source_lang": "xx"},
        headers=_auth(token),
    )
    assert resp.status_code == 400


def test_ensemble_picks_best(client: TestClient, token: str, monkeypatch) -> None:
    import backend.utils.llm_client as llm_module
    from backend.agents.critic import CriticAgent

    async def fake_generate(prompt, model=None, temperature=0.3, seed=42, max_tokens=1024):
        return f"вариант от {model}"

    monkeypatch.setattr(llm_module, "ollama_generate", fake_generate)

    scores = {"m1": 6.0, "m2": 9.0}

    async def fake_process(self, state):
        score = scores["m2"] if "m2" in state["translated_text"] else scores["m1"]
        state["critic_score"] = score
        return state

    monkeypatch.setattr(CriticAgent, "process", fake_process)

    resp = client.post(
        "/translate/ensemble",
        json={"text": "текст", "models": ["m1", "m2"]},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["variants"]) == 2
    assert body["best"]["model"] == "m2"
    assert body["best"]["critic_score"] == 9.0
