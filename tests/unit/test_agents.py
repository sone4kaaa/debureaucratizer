import pytest

from backend.agents.critic import CriticAgent, _extract_json_spans
from backend.agents.editor import EditorAgent
from backend.agents.validator import ValidatorAgent


# ---------- Validator ----------


async def test_validator_ok() -> None:
    v = ValidatorAgent()
    state = {
        "source_text": "В соответствии с договором просим произвести оплату.",
        "translated_text": "По договору нужно оплатить.",
    }
    state = await v.process(state)
    assert state["validation_passed"] is True
    assert state["validator_errors"] == []


async def test_validator_empty() -> None:
    v = ValidatorAgent()
    state = await v.process({"source_text": "текст", "translated_text": "  "})
    assert "empty_translation" in state["validator_errors"]
    assert state["validation_passed"] is False


async def test_validator_echo() -> None:
    v = ValidatorAgent()
    text = "Настоящим уведомляем о необходимости."
    state = await v.process({"source_text": text, "translated_text": text.lower()})
    assert "echo_translation" in state["validator_errors"]


async def test_validator_placeholder_mismatch() -> None:
    v = ValidatorAgent()
    state = await v.process(
        {
            "source_text": "Оплатите %1$s рублей до %s.",
            "translated_text": "Оплатите %s рублей.",
        }
    )
    assert any("placeholder_mismatch" in e for e in state["validator_errors"])


async def test_validator_cjk_error() -> None:
    v = ValidatorAgent()
    state = await v.process(
        {"source_text": "Оплатите счёт.", "translated_text": "请支付账单。"}
    )
    assert "cjk_characters_in_output" in state["validator_errors"]


async def test_validator_repeated_word_warning() -> None:
    v = ValidatorAgent()
    state = await v.process(
        {
            "source_text": "В соответствии с договором оплатите счёт до пятницы.",
            "translated_text": "По договору оплатите оплатите счёт до пятницы.",
        }
    )
    assert any("repeated_word" in w for w in state["validator_warnings"])
    assert state["validation_passed"] is True  # warning, не error


async def test_validator_bureaucratic_marker_warning() -> None:
    v = ValidatorAgent()
    state = await v.process(
        {
            "source_text": "Настоящим уведомляем о необходимости осуществления оплаты.",
            "translated_text": "Настоящим уведомляем: нужно произвести оплату.",
        }
    )
    assert any("bureaucratic_marker_left" in w for w in state["validator_warnings"])


async def test_validator_glossary_literal() -> None:
    v = ValidatorAgent()
    state = await v.process(
        {
            "source_text": "Необходимо осуществить оплату.",
            "translated_text": "Нужно оплатить.",
            "glossary_used": [
                {"source_text": "осуществить оплату", "target_text": "оплатить"}
            ],
        }
    )
    assert not any("glossary_term_missing" in w for w in state["validator_warnings"])


async def test_validator_glossary_missing() -> None:
    v = ValidatorAgent()
    state = await v.process(
        {
            "source_text": "Необходимо осуществить оплату.",
            "translated_text": "Нужно заплатить деньги.",
            "glossary_used": [
                {"source_text": "осуществить оплату", "target_text": "оплатить"}
            ],
        }
    )
    assert any("glossary_term_missing" in w for w in state["validator_warnings"])


# ---------- Critic ----------


def _critic_with_response(monkeypatch, response) -> CriticAgent:
    critic = CriticAgent(engine="ollama")

    async def fake_call(prompt: str) -> str:
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(critic, "_call_llm", fake_call)
    return critic


def test_extract_json_spans_multiple() -> None:
    text = 'текст {"a": {"b": 1}} между {"overall_score": 8} конец'
    spans = _extract_json_spans(text)
    assert len(spans) == 2
    assert '"overall_score"' in spans[1]


def test_extract_json_spans_string_with_brace() -> None:
    text = '{"note": "это } не конец", "overall_score": 7}'
    spans = _extract_json_spans(text)
    assert len(spans) == 1


async def test_critic_parses_json(monkeypatch) -> None:
    critic = _critic_with_response(
        monkeypatch,
        '{"adequacy": 9, "fluency": 8, "terminology": 9, "overall_score": 8.5, "suggestions": []}',
    )
    state = await critic.process(
        {"source_text": "a", "translated_text": "b", "domain": "general"}
    )
    assert state["critic_score"] == 8.5
    assert state["critic_passed"] is True
    assert state["critic_failed"] is False


async def test_critic_regex_fallback(monkeypatch) -> None:
    critic = _critic_with_response(
        monkeypatch, "Итог: overall_score: 6.5, больше ничего"
    )
    state = await critic.process({"source_text": "a", "translated_text": "b"})
    assert state["critic_score"] == 6.5
    assert state["critic_failed"] is False


async def test_critic_invalid_json_failed(monkeypatch) -> None:
    critic = _critic_with_response(monkeypatch, "совсем не оценка")
    state = await critic.process({"source_text": "a", "translated_text": "b"})
    assert state["critic_failed"] is True
    assert state["critic_score"] is None
    assert state["critic_passed"] is False


async def test_critic_llm_timeout(monkeypatch) -> None:
    critic = _critic_with_response(monkeypatch, TimeoutError("timeout"))
    state = await critic.process({"source_text": "a", "translated_text": "b"})
    assert state["critic_failed"] is True
    assert state["critic_score"] is None


# ---------- Editor ----------


def _editor_with_response(monkeypatch, response) -> EditorAgent:
    editor = EditorAgent(engine="ollama")

    async def fake_call(prompt: str) -> str:
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(editor, "_call_llm", fake_call)
    return editor


async def test_editor_skips_when_critic_failed(monkeypatch) -> None:
    editor = _editor_with_response(monkeypatch, "не должен вызываться")
    state = await editor.process(
        {
            "source_text": "a",
            "translated_text": "b",
            "critic_failed": True,
            "validator_errors": ["x"],
        }
    )
    assert state["editor_changed"] is False
    assert state["translated_text"] == "b"


async def test_editor_skips_without_issues(monkeypatch) -> None:
    editor = _editor_with_response(monkeypatch, "не должен вызываться")
    state = await editor.process(
        {
            "source_text": "a",
            "translated_text": "b",
            "validator_errors": [],
            "validator_warnings": [],
            "critic_suggestions": [],
        }
    )
    assert state["editor_changed"] is False
    assert state["editor_skip_reason"] == "no_issues"


async def test_editor_empty_response_keeps_original(monkeypatch) -> None:
    editor = _editor_with_response(monkeypatch, "   ")
    state = await editor.process(
        {"source_text": "a", "translated_text": "оригинал", "validator_errors": ["echo_translation"]}
    )
    assert state["editor_failed"] is True
    assert state["translated_text"] == "оригинал"


async def test_editor_applies_correction(monkeypatch) -> None:
    editor = _editor_with_response(monkeypatch, "Простой вариант.")
    state = await editor.process(
        {
            "source_text": "a",
            "translated_text": "сложный вариант",
            "validator_errors": ["echo_translation"],
        }
    )
    assert state["editor_changed"] is True
    assert state["editor_failed"] is False
    assert state["translated_text"] == "Простой вариант."
    assert state["editor_iterations"] == 1
