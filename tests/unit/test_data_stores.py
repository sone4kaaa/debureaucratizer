"""Юнит-тесты хранилищ данных: cache, glossary, TM (in-memory SQLite)."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from backend.data.cache import CacheStore
from backend.data.glossary import GlossaryStore
from backend.data.translation_memory import TMStore
from backend.db.models import TranslationCache, deterministic_id


# ---------- Cache ----------


async def test_cache_put_get_hit(test_db) -> None:
    store = CacheStore()
    async with test_db() as session:
        await store.put(
            session, "bureau", "plain", "Сложный текст", "Простой текст",
            engine="ollama", model="m1", eval_mode="full",
        )
        row = await store.get(
            session, "bureau", "plain", "Сложный текст",
            engine="ollama", model="m1", eval_mode="full",
        )
        assert row is not None
        assert row.translated_text == "Простой текст"
        assert row.hits == 1  # чтение инкрементирует счётчик


async def test_cache_miss_on_engine_mismatch(test_db) -> None:
    store = CacheStore()
    async with test_db() as session:
        await store.put(
            session, "bureau", "plain", "Текст", "Простой", engine="ollama"
        )
        assert await store.get(session, "bureau", "plain", "Текст", engine="huggingface") is None
        # None = «любой движок» — находит
        assert await store.get(session, "bureau", "plain", "Текст") is not None


async def test_cache_duplicate_put_no_error(test_db) -> None:
    store = CacheStore()
    async with test_db() as session:
        await store.put(session, "bureau", "plain", "Текст", "Один")
        # повторная вставка того же ключа — тихий rollback, без исключения
        await store.put(session, "bureau", "plain", "Текст", "Два")
        row = await store.get(session, "bureau", "plain", "Текст")
        assert row is not None


async def test_cache_clear_by_eval_mode_and_age(test_db) -> None:
    store = CacheStore()
    async with test_db() as session:
        await store.put(session, "bureau", "plain", "A", "a", eval_mode="full")
        await store.put(session, "bureau", "plain", "B", "b", eval_mode="baseline")
        deleted = await store.clear(session, eval_mode="full")
        assert deleted == 1
        assert await store.get(session, "bureau", "plain", "B") is not None

        # старение: «старее 30 дней» не трогает свежие записи
        assert await store.clear(session, older_than_days=30) == 0
        row = (
            await session.execute(select(TranslationCache))
        ).scalars().first()
        row.created_at = datetime.now(UTC) - timedelta(days=40)
        await session.commit()
        assert await store.clear(session, older_than_days=30) == 1


async def test_cache_stats(test_db) -> None:
    store = CacheStore()
    async with test_db() as session:
        await store.put(session, "bureau", "plain", "A", "a")
        await store.get(session, "bureau", "plain", "A")
        stats = await store.stats(session)
        assert stats["total_entries"] == 1
        assert stats["total_hits"] == 1


# ---------- Glossary ----------


def _entry(src: str, tgt: str) -> dict[str, str]:
    return {
        "source_lang": "bureau",
        "target_lang": "plain",
        "source_text": src,
        "target_text": tgt,
    }


async def test_glossary_add_and_deterministic_id(test_db) -> None:
    store = GlossaryStore()
    async with test_db() as session:
        entry = await store.add(session, "bureau", "plain", "в целях", "чтобы")
        assert entry.id == deterministic_id("bureau", "plain", "в целях")


async def test_glossary_add_bulk_idempotent(test_db) -> None:
    store = GlossaryStore()
    entries = [_entry("в целях", "чтобы"), _entry("осуществить", "сделать")]
    async with test_db() as session:
        assert await store.add_bulk(session, entries) == 2
        assert await store.add_bulk(session, entries) == 0  # повтор — дубликатов нет
        assert await store.count(session) == 2


async def test_glossary_list_filters_and_delete(test_db) -> None:
    store = GlossaryStore()
    async with test_db() as session:
        await store.add_bulk(session, [_entry("в целях", "чтобы")])
        rows = await store.list(session, source_lang="bureau", target_lang="plain")
        assert len(rows) == 1
        assert await store.list(session, source_lang="legal") == []
        entry_id = rows[0].id
        assert await store.delete(session, entry_id) is True
        assert await store.delete(session, entry_id) is False


async def test_glossary_find_relevant_with_cache_invalidation(test_db) -> None:
    store = GlossaryStore()
    async with test_db() as session:
        await store.add(session, "bureau", "plain", "осуществление оплаты", "оплата")
        found = await store.find_relevant(
            session, "Необходимо осуществление оплаты в срок", "bureau", "plain"
        )
        assert len(found) == 1
        assert found[0]["target_text"] == "оплата"
        # второй вызов идёт из кэша, запись добавлена после -> кэш сброшен
        await store.add(session, "bureau", "plain", "уведомляем", "сообщаем")
        found = await store.find_relevant(
            session, "Уведомляем: необходимо осуществление оплаты", "bureau", "plain"
        )
        assert len(found) == 2


# ---------- TM ----------


async def test_tm_insert_idempotent(test_db) -> None:
    store = TMStore()
    entries = [
        {"source_text": "В соответствии с вышеизложенным", "target_text": "Как сказано выше"},
        {"source_text": "Настоящим уведомляем", "target_text": "Сообщаем"},
    ]
    async with test_db() as session:
        assert await store.insert_in_session(session, entries, "bureau", "plain") == 2
        assert await store.insert_in_session(session, entries, "bureau", "plain") == 0
        assert await store.count(session) == 2


async def test_tm_empty_and_domain_default(test_db) -> None:
    store = TMStore()
    async with test_db() as session:
        assert await store.insert_in_session(session, [], "bureau", "plain") == 0
        await store.insert_in_session(
            session, [{"source_text": "A", "target_text": "a"}], "bureau", "plain"
        )
        rows = await store.get_all(session, "bureau", "plain")
        assert len(rows) == 1
        assert rows[0].domain == "general"


async def test_tm_list_filters_and_delete(test_db) -> None:
    store = TMStore()
    async with test_db() as session:
        await store.insert_in_session(
            session, [{"source_text": "A", "target_text": "a"}], "bureau", "plain"
        )
        rows = await store.list(session, source_lang="bureau")
        assert len(rows) == 1
        assert await store.list(session, source_lang="legal") == []
        assert await store.delete(session, rows[0].id) is True
        assert await store.delete(session, rows[0].id) is False
