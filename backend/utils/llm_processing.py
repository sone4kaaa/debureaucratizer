"""Очистка сырого ответа LLM: префиксы, JSON-обёртки, CJK, мета-фразы."""

import json
import re

_PREFIXES = (
    "упрощённый текст:",
    "упрощенный текст:",
    "простой вариант:",
    "перевод:",
    "исправленный текст:",
    "результат:",
    "ответ:",
    "output:",
    "translation:",
)

_EN_META_RE = re.compile(
    r"^(here is|this is|the simplified|sure[,!]|certainly[,!])", re.IGNORECASE
)
_CJK_RE = re.compile(r"[぀-ヿ一-鿿가-힯，。「」、；：]")


def clean_translation(raw: str) -> str:
    text = raw.strip()

    # JSON-обёртка: {"translation": "..."} и т.п.
    if text.startswith("{"):
        try:
            data = json.loads(text)
            for key in ("translation", "text", "result", "output", "simplified"):
                if isinstance(data, dict) and isinstance(data.get(key), str):
                    text = data[key]
                    break
        except json.JSONDecodeError:
            pass

    # Маркдаун-ограждения
    text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text, flags=re.MULTILINE).strip()

    # Известные префиксы
    lowered = text.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Английские мета-фразы в начале
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and _EN_META_RE.match(lines[0].strip()):
        lines = lines[1:]
    text = "\n".join(lines).strip()

    # Снять обрамляющие кавычки
    if len(text) >= 2 and text[0] in "\"'«" and text[-1] in "\"'»":
        text = text[1:-1].strip()

    # CJK-артефакты (китайская пунктуация от Qwen)
    text = _CJK_RE.sub(lambda m: {"，": ",", "。": ".", "；": ";", "：": ":"}.get(m.group(0), ""), text)

    return text
