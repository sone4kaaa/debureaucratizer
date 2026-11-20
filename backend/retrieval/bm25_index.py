"""BM25-индекс по стилевым парам с ленивой пересборкой при записи."""

import re
from typing import Any

from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._docs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._index: dict[tuple[str, str], BM25Okapi] = {}
        self._dirty: set[tuple[str, str]] = set()

    def add_entries(
        self, entries: list[dict[str, Any]], source_lang: str, target_lang: str
    ) -> None:
        key = (source_lang, target_lang)
        self._docs.setdefault(key, []).extend(entries)
        self._dirty.add(key)

    def _rebuild(self, key: tuple[str, str]) -> None:
        docs = self._docs.get(key, [])
        if docs:
            self._index[key] = BM25Okapi([_tokenize(d["source_text"]) for d in docs])
        else:
            self._index.pop(key, None)
        self._dirty.discard(key)

    def search(
        self, query: str, source_lang: str, target_lang: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        key = (source_lang, target_lang)
        if key in self._dirty:
            self._rebuild(key)
        index = self._index.get(key)
        if index is None:
            return []
        scores = index.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._docs[key], scores), key=lambda x: x[1], reverse=True
        )
        return [
            {"entry": doc, "score": float(score), "source": "bm25"}
            for doc, score in ranked[:top_k]
            if score > 0
        ]

    def clear(self) -> None:
        self._docs.clear()
        self._index.clear()
        self._dirty.clear()
