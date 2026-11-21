"""Юнит-тесты текстовых утилит, исключений, retry, http/llm-клиентов, метрик."""

import pytest

from backend.core.exceptions import (
    ConfigurationError,
    EngineFallbackTriggeredError,
    EngineUnavailableError,
    ModelNotLoadedError,
    RetrievalError,
    RUNGException,
    UncertainExampleNotFoundError,
    UnknownEvalModeError,
)
from backend.eval.metrics.legacy import compute_bleu, compute_chrf
from backend.eval.runner import Evaluator
from backend.utils.async_utils import retry
from backend.utils.llm_processing import clean_translation
from backend.utils.text_normalization import (
    collapse_whitespace,
    normalize_quotes,
    strip_diacritics,
)


# ---------- clean_translation ----------


def test_clean_prefix() -> None:
    assert clean_translation("Перевод: Привет, мир") == "Привет, мир"
    assert clean_translation("упрощённый текст: Всё просто") == "Всё просто"


def test_clean_json_wrapper() -> None:
    assert clean_translation('{"translation": "Привет"}') == "Привет"
    assert clean_translation('{"text": "Другой вариант"}') == "Другой вариант"


def test_clean_markdown_fence_and_meta() -> None:
    assert clean_translation("```\nПросто текст\n```") == "Просто текст"
    assert clean_translation("Sure, here is the result\nГотовый текст") == "Готовый текст"


def test_clean_quotes_and_cjk() -> None:
    assert clean_translation("«Привет»") == "Привет"
    assert clean_translation('"Привет"') == "Привет"
    # китайская пунктуация от Qwen заменяется на русскую
    assert clean_translation("Оплатите，пожалуйста。") == "Оплатите,пожалуйста."


# ---------- text_normalization ----------


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("  много   пробелов\n\tи переносов  ") == "много пробелов и переносов"


def test_strip_diacritics() -> None:
    assert strip_diacritics("ё") == "е"
    assert strip_diacritics("مُحَمَّد") == "محمد"  # арабский ташкиль


def test_normalize_quotes() -> None:
    assert normalize_quotes("«а» “b” „c“") == '"а" "b" "c"'


# ---------- exceptions ----------


def test_exception_hierarchy_and_to_dict() -> None:
    exc = RUNGException("базовая", context={"k": 1})
    assert exc.to_dict() == {"detail": "базовая", "code": "internal_error", "context": {"k": 1}}
    assert exc.status_code == 500

    cases = [
        (ConfigurationError("x"), 500, "configuration_error"),
        (UnknownEvalModeError("x"), 400, "unknown_eval_mode"),
        (EngineUnavailableError("x"), 503, "engine_unavailable"),
        (EngineFallbackTriggeredError("x"), 500, "engine_fallback_triggered"),
        (ModelNotLoadedError("x"), 503, "model_not_loaded"),
        (RetrievalError("x"), 502, "retrieval_error"),
        (UncertainExampleNotFoundError("x"), 404, "uncertain_example_not_found"),
    ]
    for exc, status, code in cases:
        assert exc.status_code == status, code
        assert exc.code == code
        assert isinstance(exc, RUNGException)


# ---------- retry ----------


async def test_retry_succeeds_after_failures() -> None:
    calls = 0

    @retry(max_attempts=3, delay=0)
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("ещё не готово")
        return "ок"

    assert await flaky() == "ок"
    assert calls == 3


async def test_retry_raises_last_error() -> None:
    calls = 0

    @retry(max_attempts=2, delay=0)
    async def always_fails() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"попытка {calls}")

    with pytest.raises(RuntimeError, match="попытка 2"):
        await always_fails()
    assert calls == 2


# ---------- http_client / llm_client ----------


async def test_http_client_per_loop_singleton() -> None:
    from backend.utils import http_client

    c1 = http_client.get_http_client()
    c2 = http_client.get_http_client()
    assert c1 is c2
    await http_client.close_all_clients()
    assert c1.is_closed
    assert not http_client._clients


async def test_ollama_generate_success_and_error(monkeypatch) -> None:
    from backend.utils import llm_client

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"response": "простой текст"}

    class _Client:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.calls: list[dict] = []

        async def post(self, url, json, timeout=None):
            self.calls.append({"timeout": timeout})
            if self.fail:
                raise ConnectionError("ollama down")
            return _Resp()

    ok_client = _Client(fail=False)
    monkeypatch.setattr(llm_client, "get_http_client", lambda timeout: ok_client)
    assert await llm_client.ollama_generate("промпт") == "простой текст"
    # регрессия: таймаут задаётся на запрос, а не на общий per-loop клиент
    assert ok_client.calls[0]["timeout"] == 300.0

    monkeypatch.setattr(llm_client, "get_http_client", lambda timeout: _Client(fail=True))
    with pytest.raises(EngineUnavailableError):
        await llm_client.ollama_generate("промпт")


# ---------- metrics ----------


def test_bleu_chrf_identical() -> None:
    hyps = ["Оплатите счёт до пятницы"]
    refs = ["Оплатите счёт до пятницы"]
    assert compute_bleu(hyps, refs) == pytest.approx(100.0)
    assert compute_chrf(hyps, refs) > 90.0


def test_evaluator_validation_and_modes() -> None:
    ev = Evaluator()
    with pytest.raises(ValueError):
        ev.run([], [])
    with pytest.raises(ValueError):
        ev.run(["a"], ["a", "b"])

    result = ev.run(["простой текст"], ["простой текст"], include_neural=False)
    assert result["count"] == 1
    assert "bleu" in result and "chrf++" in result
    assert "bertscore" not in result

    # neural-метрики опциональны: ошибки не роняют прогон
    result = ev.run(["простой текст"], ["простой текст"], include_neural=True)
    assert "bleu" in result
