import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db.models import (
    Base,
    GlossaryDB,
    TMDB,
    TranslationCache,
    UncertainExample,
    User,
    deterministic_id,
)


@pytest.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def test_all_tables_created(session) -> None:
    tables = set(Base.metadata.tables)
    assert tables == {
        "users",
        "glossaries",
        "translation_memories",
        "uncertain_examples",
        "translation_cache",
    }


async def test_deterministic_id() -> None:
    a = deterministic_id("bureau", "plain", "в соответствии с")
    b = deterministic_id("bureau", "plain", "в соответствии с")
    c = deterministic_id("bureau", "plain", "другой текст")
    assert a == b
    assert a != c
    assert a.startswith("bureau|plain|")


async def test_insert_and_select_glossary(session) -> None:
    entry = GlossaryDB(
        id=deterministic_id("bureau", "plain", "осуществлять"),
        source_lang="bureau",
        target_lang="plain",
        source_text="осуществлять",
        target_text="делать",
    )
    session.add(entry)
    await session.commit()

    rows = (await session.execute(select(GlossaryDB))).scalars().all()
    assert len(rows) == 1
    assert rows[0].target_text == "делать"


async def test_insert_all_models(session) -> None:
    session.add(
        TMDB(
            id=deterministic_id("bureau", "plain", "сложный текст"),
            source_lang="bureau",
            target_lang="plain",
            source_text="сложный текст",
            target_text="простой текст",
        )
    )
    session.add(
        User(
            email="a@b.c",
            username="admin",
            hashed_password="hash",
            role="admin",
        )
    )
    session.add(
        UncertainExample(
            source_lang="bureau",
            target_lang="plain",
            source_text="уведомляем",
            machine_translation="сообщаем",
        )
    )
    session.add(
        TranslationCache(
            source_lang="bureau",
            target_lang="plain",
            source_text="текст",
            translated_text="простой текст",
        )
    )
    await session.commit()

    assert (await session.execute(select(TMDB))).scalars().first() is not None
    assert (await session.execute(select(User))).scalars().first().role == "admin"
    assert (
        (await session.execute(select(UncertainExample))).scalars().first().status
        == "pending"
    )
    cache_row = (await session.execute(select(TranslationCache))).scalars().first()
    assert cache_row.hits == 0
    assert cache_row.engine_used == "unknown"
