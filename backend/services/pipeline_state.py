"""Общие синглтоны пайплайна: BM25-индекс и гибридный retriever."""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.data.embeddings import EmbeddingModel
from backend.data.translation_memory import tm_store
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever

_bm25: BM25Index | None = None
_hybrid: HybridRetriever | None = None
_hybrid_loaded_pairs: set[tuple[str, str]] = set()


def get_bm25_index() -> BM25Index:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index()
    return _bm25


async def get_hybrid_retriever(session: AsyncSession) -> HybridRetriever:
    global _hybrid
    if _hybrid is None:
        from backend.retrieval.vector_index import VectorIndex

        try:
            vector = VectorIndex()
            if not vector.ping():
                vector = None
        except Exception:
            vector = None
        _hybrid = HybridRetriever(get_bm25_index(), vector, EmbeddingModel())
    return _hybrid


async def ensure_indexes_loaded(session: AsyncSession) -> None:
    """Загружает TM из БД в BM25/Qdrant при старте (без drop коллекции)."""
    from backend.db.models import TMDB
    from sqlalchemy import select

    pairs = (await session.execute(select(TMDB.source_lang, TMDB.target_lang).distinct())).all()
    bm25 = get_bm25_index()
    for source_lang, target_lang in pairs:
        if (source_lang, target_lang) in _hybrid_loaded_pairs:
            continue
        rows = await tm_store.get_all(session, source_lang, target_lang)
        entries = [
            {"source_text": r.source_text, "target_text": r.target_text} for r in rows
        ]
        if entries:
            bm25.add_entries(entries, source_lang, target_lang)
            retriever = await get_hybrid_retriever(session)
            if retriever._vector is not None:
                vectors = await EmbeddingModel().encode_async(
                    [e["source_text"] for e in entries]
                )
                await retriever._vector.add_entries(
                    entries, vectors, source_lang, target_lang
                )
        _hybrid_loaded_pairs.add((source_lang, target_lang))
