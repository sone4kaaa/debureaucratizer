"""Rule-based валидатор упрощённого текста. LLM не вызывает."""

import re
from typing import Any

from backend.agents.base import BaseAgent
from backend.core.constants import GLOSSARY_SIMILARITY_THRESHOLD

# Маркеры канцелярита: в упрощённом тексте их быть не должно.
BUREAUCRATIC_MARKERS: tuple[str, ...] = (
    "в соответствии с",
    "вышеизложенн",
    "вышеуказанн",
    "вышеупомянут",
    "осуществля",
    "осуществлени",
    "надлежащим образом",
    "надлежащ",
    "в целях",
    "во избежание",
    "довести до сведения",
    "уведомля",
    "извеща",
    "принимая во внимание",
    "в связи с вышеизложенным",
    "произвести оплату",
    "является невозможным",
    "не позднее",
    "в установленном порядке",
    "по факту",
    "в случае неисполнения",
)

_WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[぀-ヿ一-鿿가-힯]")
_PLACEHOLDER_RE = re.compile(r"%\d*\$?[sd]")
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)

_STOPWORDS = {
    "это", "как", "или", "для", "что", "все", "его", "уже", "при", "был",
    "она", "они", "был", "так", "вам", "вас", "тот", "где", "там", "нет",
}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class ValidatorAgent(BaseAgent):
    def __init__(self, engine: str | None = None) -> None:
        # engine принимается для единообразной сборки в Supervisor, но не используется
        super().__init__("validator")

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        source: str = state.get("source_text", "")
        translation: str = state.get("translated_text", "")
        errors: list[str] = []
        warnings: list[str] = []

        # Пустой результат
        if not translation.strip():
            errors.append("empty_translation")

        # Эхо: результат совпадает со входом после нормализации
        if translation.strip() and _normalize(translation) == _normalize(source):
            errors.append("echo_translation")

        # Плейсхолдеры %s / %1$s — мультимножества должны совпадать
        src_ph = sorted(_PLACEHOLDER_RE.findall(source))
        tgt_ph = sorted(_PLACEHOLDER_RE.findall(translation))
        if src_ph != tgt_ph:
            errors.append(
                f"placeholder_mismatch: источник {src_ph}, результат {tgt_ph}"
            )

        # Глоссарий: термины из glossary_used, встречающиеся в источнике,
        # должны быть заменены простым эквивалентом или сохранены (если термин
        # обязателен). Для дебюрократизатора: канцеляризм в источнике должен
        # исчезнуть, его простой эквивалент — появиться.
        for entry in state.get("glossary_used") or []:
            src_term = entry.get("source_text", "")
            tgt_term = entry.get("target_text", "")
            if not src_term or _normalize(src_term) not in _normalize(source):
                continue
            if _normalize(tgt_term) in _normalize(translation):
                continue
            if len(tgt_term) >= 5 and await self._semantic_match(tgt_term, translation):
                continue
            warnings.append(f"glossary_term_missing: {src_term!r} -> {tgt_term!r}")

        # Канцеляризмы не должны остаться в упрощённом тексте
        lowered = translation.lower()
        for marker in BUREAUCRATIC_MARKERS:
            if marker in lowered:
                warnings.append(f"bureaucratic_marker_left: {marker!r}")

        # Скрипт целевого языка: простой русский — кириллица >= 50% букв
        if translation.strip():
            cyr = len(_CYRILLIC_RE.findall(translation))
            lat = len(_LATIN_RE.findall(translation))
            if cyr + lat > 0 and cyr / (cyr + lat) < 0.5:
                errors.append("wrong_script: доля кириллицы < 50%")

        # CJK в русском тексте — ошибка
        if _CJK_RE.search(translation):
            errors.append("cjk_characters_in_output")

        # Повтор подряд идущих слов
        words = _words(translation)
        for i in range(1, len(words)):
            if words[i] == words[i - 1] and len(words[i]) > 2:
                warnings.append(f"repeated_word: {words[i]!r}")
                break

        # Длинные фрагменты, скопированные без переработки (>= 4 слов подряд)
        leaked = self._verbatim_overlap(source, translation)
        for frag in leaked[:5]:
            warnings.append(f"unprocessed_fragment: {frag!r}")

        result = {
            "validator_errors": errors,
            "validator_warnings": warnings,
            "validation_passed": not errors,
        }
        state.update(result)
        self._last_result = result
        return state

    async def _semantic_match(self, term: str, translation: str) -> bool:
        """Термин считается присутствующим, если embedding similarity >= 0.7."""
        try:
            from backend.data.embeddings import EmbeddingModel

            model = EmbeddingModel()
            vec_term, vec_text = model.encode([term, translation])
            sim = model.cosine_similarity(vec_term, vec_text)
            return sim >= GLOSSARY_SIMILARITY_THRESHOLD
        except Exception:
            return False

    @staticmethod
    def _verbatim_overlap(source: str, translation: str, n: int = 4) -> list[str]:
        """N-граммы источника, дословно вошедшие в результат (кроме чисел/дат)."""
        src_words = _words(source)
        tgt_words = set(_words(translation))
        fragments: list[str] = []
        for i in range(len(src_words) - n + 1):
            ngram = src_words[i: i + n]
            if any(len(w) < 4 or w in _STOPWORDS for w in ngram):
                continue
            if all(w in tgt_words for w in ngram):
                # дополнительная проверка: подряд ли они в результате
                phrase = " ".join(ngram)
                if phrase in " ".join(_words(translation)):
                    fragments.append(phrase)
        return fragments
