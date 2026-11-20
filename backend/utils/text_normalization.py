import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WORD_JOIN(text)


def _WORD_JOIN(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def strip_diacritics(text: str) -> str:
    """Удаляет комбинируемые диакритические знаки (в т.ч. арабские ташкиль)."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def normalize_quotes(text: str) -> str:
    return (
        text.replace("«", '"')
        .replace("»", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
    )
