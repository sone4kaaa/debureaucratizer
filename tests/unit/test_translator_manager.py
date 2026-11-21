"""Юнит-тесты TranslatorManager: выбор движка, fallback, strict_engine."""

import pytest

from backend.core.config import get_settings
from backend.core.exceptions import (
    EngineFallbackTriggeredError,
    EngineUnavailableError,
)
from backend.data import translator_manager as tm_module
from backend.data.translator_manager import TranslatorManager


def _manager(monkeypatch, ollama_ok: bool) -> TranslatorManager:
    manager = TranslatorManager()

    async def fake_probe() -> bool:
        return ollama_ok

    monkeypatch.setattr(manager, "_is_ollama_available", fake_probe)
    return manager


async def test_requested_huggingface_skips_probe(monkeypatch) -> None:
    manager = TranslatorManager()

    async def forbidden_probe() -> bool:
        raise AssertionError("проба Ollama не должна вызываться")

    monkeypatch.setattr(manager, "_is_ollama_available", forbidden_probe)
    assert await manager.choose_engine("bureau", "plain", "huggingface") == "huggingface"


async def test_ollama_ok_known_pair(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=True)
    assert await manager.choose_engine("bureau", "plain", "ollama") == "ollama"
    assert await manager.choose_engine("bureau", "plain", "auto") == "ollama"


async def test_ollama_down_fallback_to_hf(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=False)
    settings = get_settings()
    monkeypatch.setattr(settings, "FALLBACK_TO_HF_ON_ERROR", True)
    assert await manager.choose_engine("bureau", "plain", "ollama") == "huggingface"
    assert await manager.choose_engine("bureau", "plain", "auto") == "huggingface"


async def test_ollama_down_fallback_disabled_raises(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=False)
    settings = get_settings()
    monkeypatch.setattr(settings, "FALLBACK_TO_HF_ON_ERROR", False)
    with pytest.raises(EngineUnavailableError):
        await manager.choose_engine("bureau", "plain", "ollama")


async def test_unknown_pair_falls_back(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=True)
    settings = get_settings()
    monkeypatch.setattr(settings, "FALLBACK_TO_HF_ON_ERROR", True)
    assert await manager.choose_engine("xx", "plain", "ollama") == "huggingface"


async def test_translate_via_ollama(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=True)

    async def fake_generate(prompt, model=None, temperature=0.3, seed=42, max_tokens=1024):
        return "Перевод: Оплатите счёт до пятницы"

    monkeypatch.setattr(tm_module, "ollama_generate", fake_generate)
    result = await manager.translate("промпт", "bureau", "plain", engine="ollama")
    assert result["text"] == "Оплатите счёт до пятницы"  # clean_translation сработал
    assert result["engine_used"] == "ollama"
    assert result["model_used"] == get_settings().OLLAMA_MODEL


async def test_strict_engine_raises_on_fallback(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=False)
    settings = get_settings()
    monkeypatch.setattr(settings, "FALLBACK_TO_HF_ON_ERROR", True)
    with pytest.raises(EngineFallbackTriggeredError):
        await manager.translate(
            "промпт", "bureau", "plain", engine="ollama", strict_engine=True
        )


async def test_translate_via_huggingface(monkeypatch) -> None:
    manager = _manager(monkeypatch, ollama_ok=False)

    class _FakeHF:
        async def translate(self, prompt: str) -> str:
            return "Простой текст от HF"

    import backend.data.huggingface_translator as hf_module

    monkeypatch.setattr(hf_module, "TranslatorHuggingFace", _FakeHF)
    result = await manager.translate("промпт", "bureau", "plain", engine="huggingface")
    assert result["text"] == "Простой текст от HF"
    assert result["engine_used"] == "huggingface"
    assert result["model_used"] == get_settings().HF_MODEL
