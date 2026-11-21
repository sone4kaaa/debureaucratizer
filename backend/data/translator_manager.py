"""Выбор бэкенда упрощения: Ollama (основной) / HuggingFace (запасной)."""

import time

from backend.core.config import get_settings
from backend.core.constants import DEFAULT_SEED, KNOWN_STYLES
from backend.core.exceptions import EngineFallbackTriggeredError, EngineUnavailableError
from backend.utils.http_client import get_http_client
from backend.utils.llm_client import ollama_generate
from backend.utils.llm_processing import clean_translation
from backend.utils.logging import logger

_OLLAMA_PROBE_TTL = 30.0


class TranslatorManager:
    def __init__(self) -> None:
        self._ollama_available: bool | None = None
        self._ollama_probe_ts: float = 0.0
        self._hf = None

    async def _is_ollama_available(self) -> bool:
        now = time.monotonic()
        if self._ollama_available is not None and now - self._ollama_probe_ts < _OLLAMA_PROBE_TTL:
            return self._ollama_available
        settings = get_settings()
        try:
            client = get_http_client(timeout=5.0)
            resp = await client.get(f"{settings.ollama_url}/api/tags", timeout=5.0)
            self._ollama_available = resp.status_code == 200
        except Exception:
            self._ollama_available = False
        self._ollama_probe_ts = now
        return self._ollama_available

    async def choose_engine(
        self, source_lang: str, target_lang: str, requested: str | None = None
    ) -> str:
        settings = get_settings()
        requested = requested or settings.TRANSLATION_ENGINE
        if requested == "huggingface":
            return "huggingface"
        pair_known = source_lang in KNOWN_STYLES and target_lang in KNOWN_STYLES
        ollama_ok = await self._is_ollama_available()
        if requested == "ollama":
            if ollama_ok and pair_known:
                return "ollama"
            if settings.FALLBACK_TO_HF_ON_ERROR:
                logger.warning("Ollama недоступна/пара неизвестна, fallback на HF")
                return "huggingface"
            raise EngineUnavailableError("Ollama недоступна, fallback отключён")
        # auto
        if ollama_ok and pair_known:
            return "ollama"
        return "huggingface"

    async def translate(
        self,
        prompt: str,
        source_lang: str,
        target_lang: str,
        engine: str | None = None,
        strict_engine: bool = False,
    ) -> dict:
        """Возвращает {text, engine_used, model_used}."""
        settings = get_settings()
        chosen = await self.choose_engine(source_lang, target_lang, engine)
        if strict_engine and (engine or settings.TRANSLATION_ENGINE) == "ollama" and chosen != "ollama":
            raise EngineFallbackTriggeredError(
                "Запрошен ollama, но выбран HF — eval-режим деградировал бы в baseline"
            )

        if chosen == "ollama":
            raw = await ollama_generate(
                prompt, model=settings.OLLAMA_MODEL, temperature=0.3, seed=DEFAULT_SEED
            )
            return {
                "text": clean_translation(raw),
                "engine_used": "ollama",
                "model_used": settings.OLLAMA_MODEL,
            }

        from backend.data.huggingface_translator import TranslatorHuggingFace

        if self._hf is None:
            self._hf = TranslatorHuggingFace()
        raw = await self._hf.translate(prompt)
        return {
            "text": clean_translation(raw),
            "engine_used": "huggingface",
            "model_used": get_settings().HF_MODEL,
        }


translator_manager = TranslatorManager()
