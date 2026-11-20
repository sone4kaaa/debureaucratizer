"""Гибридный поиск: BM25 + векторный, слияние Reciprocal Rank Fusion (k=60)."""

from typing import Any

from backend.data.embeddings import EmbeddingModel
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_index import VectorIndex

RRF_K = 60


def _entry_key(entry: dict[str, Any], source_lang: str, target_lang: str) -> tuple:
    return (
        entry.get("source_text", ""),
        entry.get("target_text", ""),
        source_lang,
        target_lang,
    )


def rrf_merge(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    source_lang: str,
    target_lang: str,
    k: int = RRF_K,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """RRF: score(entry) = Σ_rank 1 / (k + rank + 1), с дедупликацией."""
    scores: dict[tuple, float] = {}
    entries: dict[tuple, dict[str, Any]] = {}
    for results in (bm25_results, vector_results):
        for rank, item in enumerate(results):
            key = _entry_key(item["entry"], source_lang, target_lang)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            entries.setdefault(key, item["entry"])
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"entry": entries[key], "score": score, "source": "hybrid"}
        for key, score in ranked[:top_k]
    ]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        vector: VectorIndex | None,
        embeddings: EmbeddingModel,
    ) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._embeddings = embeddings

    async def search(
        self, query: str, source_lang: str, target_lang: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        bm25_results = self._bm25.search(query, source_lang, target_lang, top_k=top_k)
        vector_results: list[dict[str, Any]] = []
        if self._vector is not None:
            vec = (await self._embeddings.encode_async([query]))[0]
            vector_results = await self._vector.search(
                vec, source_lang, target_lang, top_k=top_k
            )
        return rrf_merge(bm25_results, vector_results, source_lang, target_lang, top_k=top_k)
