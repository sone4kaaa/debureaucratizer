"""База знаний пар «сложное -> простое» (TMDB) с идемпотентной вставкой."""

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import TMDB, deterministic_id


def _insert_on_conflict_do_nothing(session: AsyncSession):
    engine_name = session.bind.dialect.name if session.bind else "sqlite"
    if engine_name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    return sqlite_insert


class TMStore:
    async def insert_in_session(
        self,
        session: AsyncSession,
        entries: list[dict[str, str]],
        source_lang: str,
        target_lang: str,
        domain: str = "general",
    ) -> int:
        """INSERT ... ON CONFLICT (id) DO NOTHING: повторный импорт не плодит дубли."""
        if not entries:
            return 0
        insert = _insert_on_conflict_do_nothing(session)
        rows = [
            {
                "id": deterministic_id(source_lang, target_lang, e["source_text"]),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": e["source_text"],
                "target_text": e["target_text"],
                "domain": e.get("domain", domain),
            }
            for e in entries
        ]
        result = await session.execute(
            insert(TMDB).values(rows).on_conflict_do_nothing(index_elements=["id"])
        )
        await session.commit()
        return result.rowcount if result.rowcount is not None else 0

    async def get_all(
        self, session: AsyncSession, source_lang: str, target_lang: str
    ) -> list[TMDB]:
        stmt = select(TMDB).where(
            TMDB.source_lang == source_lang, TMDB.target_lang == target_lang
        )
        return list((await session.execute(stmt)).scalars().all())

    async def list(
        self,
        session: AsyncSession,
        source_lang: str | None = None,
        target_lang: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TMDB]:
        stmt = select(TMDB)
        if source_lang:
            stmt = stmt.where(TMDB.source_lang == source_lang)
        if target_lang:
            stmt = stmt.where(TMDB.target_lang == target_lang)
        stmt = stmt.limit(limit).offset(offset)
        return list((await session.execute(stmt)).scalars().all())

    async def count(self, session: AsyncSession) -> int:
        return (await session.execute(select(func.count(TMDB.id)))).scalar_one()

    async def delete(self, session: AsyncSession, entry_id: str) -> bool:
        result = await session.execute(delete(TMDB).where(TMDB.id == entry_id))
        await session.commit()
        return result.rowcount > 0


tm_store = TMStore()
