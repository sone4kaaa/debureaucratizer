"""Фоновые задачи Celery."""

from typing import Any

from backend.utils.logging import logger
from backend.worker.celery_app import celery_app, get_worker_loop


def _run(coro):
    return get_worker_loop().run_until_complete(coro)


async def _session_scope():
    from backend.db.session import AsyncSessionLocal

    return AsyncSessionLocal()


@celery_app.task(name="index_translation_memory")
def index_translation_memory(source_lang: str = "bureau", target_lang: str = "plain") -> dict[str, Any]:
    """Полная переиндексация TM: BM25 + пересоздание коллекции Qdrant."""

    async def _job() -> dict[str, Any]:
        from backend.data.embeddings import EmbeddingModel
        from backend.data.translation_memory import tm_store
        from backend.retrieval.vector_index import VectorIndex
        from backend.services.pipeline_state import get_bm25_index

        async with await _session_scope() as session:
            rows = await tm_store.get_all(session, source_lang, target_lang)
        entries = [{"source_text": r.source_text, "target_text": r.target_text} for r in rows]
        if not entries:
            return {"indexed": 0}

        bm25 = get_bm25_index()
        bm25.add_entries(entries, source_lang, target_lang)

        vector = VectorIndex()
        vector.recreate_collection()
        vectors = await EmbeddingModel().encode_async([e["source_text"] for e in entries])
        await vector.add_entries(entries, vectors, source_lang, target_lang)
        return {"indexed": len(entries)}

    result = _run(_job())
    logger.info("index_translation_memory: %s", result)
    return result


@celery_app.task(name="batch_translate")
def batch_translate(
    texts: list[str], source_lang: str = "bureau", target_lang: str = "plain"
) -> list[dict[str, Any]]:
    async def _job() -> list[dict[str, Any]]:
        import asyncio

        from backend.services.pipeline import perform_translation

        sem = asyncio.Semaphore(4)
        async with await _session_scope() as session:

            async def _one(text: str) -> dict[str, Any]:
                async with sem:
                    return await perform_translation(
                        session, text, source_lang=source_lang, target_lang=target_lang
                    )

            return await asyncio.gather(*(_one(t) for t in texts))

    return _run(_job())


@celery_app.task(name="run_evaluation_async")
def run_evaluation_async(
    hypotheses: list[str], references: list[str], include_neural: bool = False
) -> dict[str, Any]:
    from backend.eval.runner import evaluator

    return evaluator.run(hypotheses, references, include_neural=include_neural)


@celery_app.task(name="clear_old_cache")
def clear_old_cache(days: int = 30) -> dict[str, int]:
    async def _job() -> int:
        from backend.data.cache import cache_store

        async with await _session_scope() as session:
            return await cache_store.clear(session, older_than_days=days)

    return {"deleted": _run(_job())}


@celery_app.task(name="add_glossary_bulk")
def add_glossary_bulk(entries: list[dict[str, str]]) -> dict[str, int]:
    async def _job() -> int:
        from backend.data.glossary import glossary_store

        async with await _session_scope() as session:
            return await glossary_store.add_bulk(session, entries)

    return {"added": _run(_job())}


@celery_app.task(name="add_tm_bulk")
def add_tm_bulk(
    entries: list[dict[str, str]], source_lang: str = "bureau", target_lang: str = "plain"
) -> dict[str, Any]:
    async def _job() -> int:
        from backend.data.translation_memory import tm_store

        async with await _session_scope() as session:
            return await tm_store.insert_in_session(session, entries, source_lang, target_lang)

    added = _run(_job())
    # После массового импорта — переиндексация
    index_translation_memory.delay(source_lang, target_lang)
    return {"added": added, "reindex_started": True}
