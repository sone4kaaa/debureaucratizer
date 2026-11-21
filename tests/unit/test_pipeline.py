"""Юнит-тесты пайплайна: режимы, build_prompt, perform_translation с моками."""

import pytest
from sqlalchemy import func, select

import backend.services.pipeline as pipeline
from backend.core.exceptions import UnknownEvalModeError
from backend.db.models import UncertainExample
from backend.services.pipeline import build_prompt, get_mode_config, perform_translation
from backend.services.pipeline_state import get_bm25_index


class _FakeManager:
    """Замена TranslatorManager: возвращает фиксированный «перевод»."""

    def __init__(self, text: str = "Оплатите счёт до 15 марта.") -> None:
        self.text = text
        self.prompts: list[str] = []

    async def translate(
        self, prompt, source_lang, target_lang, engine=None, strict_engine=False
    ) -> dict:
        self.prompts.append(prompt)
        return {"text": self.text, "engine_used": "ollama", "model_used": "test-model"}


# ---------- режимы и промпт ----------


def test_get_mode_config_all_modes() -> None:
    assert len(pipeline.EVAL_MODES) == 8
    cfg = get_mode_config("baseline")
    assert cfg == {"glossary": False, "tm": None, "critic": False, "editor": False}
    cfg = get_mode_config("full")
    assert cfg["tm"] == "hybrid" and cfg["critic"] and cfg["editor"]


def test_get_mode_config_unknown() -> None:
    with pytest.raises(UnknownEvalModeError) as exc_info:
        get_mode_config("нет-такого")
    assert exc_info.value.status_code == 400
    assert "modes" in exc_info.value.context


def test_build_prompt_with_glossary_and_tm() -> None:
    glossary = [
        {"source_text": f"термин{i}", "target_text": f"простое{i}"} for i in range(25)
    ]
    tm = [
        {"entry": {"source_text": "сложное", "target_text": "простое"}, "score": 0.75}
    ]
    prompt = build_prompt("Исходный текст", glossary, tm)
    assert "Исходный текст" in prompt
    assert "термин19" in prompt  # глоссарий обрезан до 20 строк
    assert "термин20" not in prompt
    assert "«сложное» → «простое» (score: 0.75)" in prompt


def test_build_prompt_minimal() -> None:
    prompt = build_prompt("Только текст", [], [])
    assert "Замени канцеляризмы" not in prompt
    assert "Примеры упрощения" not in prompt
    assert "Только текст" in prompt


# ---------- perform_translation ----------


async def test_perform_translation_baseline_and_cache(test_db, monkeypatch) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(pipeline, "translator_manager", manager)
    source = "В соответствии с договором необходимо осуществить оплату счёта в срок до 15 марта."

    async with test_db() as session:
        result = await perform_translation(session, source, eval_mode="baseline")
        assert result["translated_text"] == "Оплатите счёт до 15 марта."
        assert result["from_cache"] is False
        assert result["stopped_reason"] == "agents_disabled"
        assert result["final_success"] is True
        assert result["eval_mode"] == "baseline"

        # второй вызов — из кэша
        cached = await perform_translation(session, source, eval_mode="baseline")
        assert cached["from_cache"] is True
        assert cached["translated_text"] == result["translated_text"]
        assert cached["engine_used"] == "ollama"
        assert len(manager.prompts) == 1  # LLM повторно не вызывалась


async def test_perform_translation_no_cache_for_other_domain(
    test_db, monkeypatch
) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(pipeline, "translator_manager", manager)
    async with test_db() as session:
        result = await perform_translation(
            session, "Канцелярский текст", domain="legal", eval_mode="baseline"
        )
        assert result["from_cache"] is False
        again = await perform_translation(
            session, "Канцелярский текст", domain="legal", eval_mode="baseline"
        )
        assert again["from_cache"] is False
        assert len(manager.prompts) == 2


async def test_perform_translation_tm_mode_uses_bm25(test_db, monkeypatch) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(pipeline, "translator_manager", manager)
    get_bm25_index().clear()
    # >=3 документа: иначе idf общих терминов уходит в ноль/минус
    get_bm25_index().add_entries(
        [
            {
                "source_text": "осуществить оплату в установленный срок",
                "target_text": "оплатить вовремя",
            },
            {"source_text": "направить уведомление заявителю", "target_text": "сообщить"},
            {"source_text": "произвести возврат денежных средств", "target_text": "вернуть"},
        ],
        "bureau",
        "plain",
    )
    async with test_db() as session:
        result = await perform_translation(
            session, "Просим осуществить оплату", eval_mode="tm"
        )
        assert result["final_success"] is True
        assert "Примеры упрощения" in manager.prompts[0]
    get_bm25_index().clear()


async def test_perform_translation_failure_pushes_to_hitl(
    test_db, monkeypatch
) -> None:
    source = "Текст без изменений"
    manager = _FakeManager(text=source)  # эхо -> validator error
    monkeypatch.setattr(pipeline, "translator_manager", manager)
    async with test_db() as session:
        result = await perform_translation(session, source, eval_mode="baseline")
        assert result["final_success"] is False
        assert "echo_translation" in result["validator_errors"]
        count = (
            await session.execute(select(func.count(UncertainExample.id)))
        ).scalar_one()
        assert count == 1  # неоднозначный результат попал в HITL-очередь


async def test_perform_translation_unknown_mode(test_db) -> None:
    async with test_db() as session:
        with pytest.raises(UnknownEvalModeError):
            await perform_translation(session, "текст", eval_mode="неверный")
