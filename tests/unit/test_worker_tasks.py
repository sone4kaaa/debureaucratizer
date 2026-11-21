"""Тесты Celery-задач (синхронные: worker использует собственный event loop)."""

import types

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.services.pipeline as pipeline
import backend.worker.tasks as tasks
from backend.db.models import Base
from backend.worker.celery_app import _on_worker_shutdown, get_worker_loop


@pytest.fixture()
def worker_db(monkeypatch):
    """In-memory SQLite, подменяющая AsyncSessionLocal в задачах."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    loop = get_worker_loop()

    async def _create() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop.run_until_complete(_create())
    monkeypatch.setattr("backend.db.session.AsyncSessionLocal", factory)
    yield factory
    loop.run_until_complete(engine.dispose())


def test_clear_old_cache(worker_db) -> None:
    from backend.data.cache import cache_store

    async def _seed() -> None:
        async with worker_db() as session:
            await cache_store.put(session, "bureau", "plain", "текст", "простой")

    get_worker_loop().run_until_complete(_seed())
    assert tasks.clear_old_cache(days=0) == {"deleted": 1}
    assert tasks.clear_old_cache(days=30) == {"deleted": 0}


def test_add_glossary_bulk(worker_db) -> None:
    entries = [
        {
            "source_lang": "bureau",
            "target_lang": "plain",
            "source_text": "в целях",
            "target_text": "чтобы",
        }
    ]
    assert tasks.add_glossary_bulk(entries) == {"added": 1}
    assert tasks.add_glossary_bulk(entries) == {"added": 0}  # идемпотентно


def test_add_tm_bulk_triggers_reindex(worker_db, monkeypatch) -> None:
    calls: list[tuple] = []
    fake_task = types.SimpleNamespace(delay=lambda *a: calls.append(a))
    monkeypatch.setattr(tasks, "index_translation_memory", fake_task)

    result = tasks.add_tm_bulk(
        [{"source_text": "сложное", "target_text": "простое"}], "bureau", "plain"
    )
    assert result == {"added": 1, "reindex_started": True}
    assert calls == [("bureau", "plain")]


def test_run_evaluation_async() -> None:
    # BLEU без сглаживания зануляется на текстах короче 4 слов
    text = "Оплатите счёт до пятницы вечера"
    result = tasks.run_evaluation_async([text], [text], include_neural=False)
    assert result["count"] == 1
    assert result["bleu"] == pytest.approx(100.0)
    assert "chrf++" in result


def test_batch_translate(worker_db, monkeypatch) -> None:
    async def fake_perform(session, text, source_lang="bureau", target_lang="plain", **kw):
        return {"source_text": text, "translated_text": f"простой: {text}"}

    monkeypatch.setattr(pipeline, "perform_translation", fake_perform)
    results = tasks.batch_translate(["один", "два", "три"])
    assert len(results) == 3
    assert results[0]["translated_text"] == "простой: один"


def test_index_translation_memory(worker_db, monkeypatch) -> None:
    rows = [
        types.SimpleNamespace(source_text="сложное один", target_text="простое один"),
        types.SimpleNamespace(source_text="сложное два", target_text="простое два"),
    ]

    async def fake_get_all(session, source_lang, target_lang):
        return rows

    from backend.data.translation_memory import tm_store

    monkeypatch.setattr(tm_store, "get_all", fake_get_all)

    from backend.retrieval.bm25_index import BM25Index
    import backend.services.pipeline_state as state_module

    monkeypatch.setattr(state_module, "get_bm25_index", BM25Index)

    added: dict[str, int] = {}

    class _FakeVector:
        def recreate_collection(self) -> None:
            pass

        async def add_entries(self, entries, vectors, source_lang, target_lang) -> None:
            added["count"] = len(entries)

    import backend.retrieval.vector_index as vector_module

    monkeypatch.setattr(vector_module, "VectorIndex", _FakeVector)

    class _FakeEmb:
        async def encode_async(self, texts):
            return [[0.0, 0.1] for _ in texts]

    import backend.data.embeddings as emb_module

    monkeypatch.setattr(emb_module, "EmbeddingModel", _FakeEmb)

    result = tasks.index_translation_memory("bureau", "plain")
    assert result == {"indexed": 2}
    assert added["count"] == 2


def test_index_translation_memory_empty(worker_db, monkeypatch) -> None:
    async def fake_get_all(session, source_lang, target_lang):
        return []

    from backend.data.translation_memory import tm_store

    monkeypatch.setattr(tm_store, "get_all", fake_get_all)
    assert tasks.index_translation_memory("bureau", "plain") == {"indexed": 0}


def test_worker_loop_recreated_after_shutdown() -> None:
    _on_worker_shutdown()  # закрывает текущий loop
    loop = get_worker_loop()
    assert loop is not None and not loop.is_closed()
