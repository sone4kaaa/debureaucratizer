"""Справочник канцеляризмов (GlossaryDB) с дедупликацией по SHA-id."""

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import GlossaryDB, deterministic_id


class GlossaryStore:
    def __init__(self) -> None:
        self._query_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    async def add(
        self,
        session: AsyncSession,
        source_lang: str,
        target_lang: str,
        source_text: str,
        target_text: str,
    ) -> GlossaryDB:
        entry = GlossaryDB(
            id=deterministic_id(source_lang, target_lang, source_text),
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            target_text=target_text,
        )
        session.add(entry)
        await session.commit()
        self._query_cache.clear()
        return entry

    async def add_bulk(
        self, session: AsyncSession, entries: list[dict[str, str]]
    ) -> int:
        """entries: [{source_lang, target_lang, source_text, target_text}]. Идемпотентно."""
        added = 0
        for e in entries:
            entry_id = deterministic_id(e["source_lang"], e["target_lang"], e["source_text"])
            exists = (
                await session.execute(
                    select(GlossaryDB.id).where(GlossaryDB.id == entry_id)
                )
            ).scalar_one_or_none()
            if exists is None:
                session.add(
                    GlossaryDB(
                        id=entry_id,
                        source_lang=e["source_lang"],
                        target_lang=e["target_lang"],
                        source_text=e["source_text"],
                        target_text=e["target_text"],
                    )
                )
                added += 1
        await session.commit()
        self._query_cache.clear()
        return added

    async def list(
        self,
        session: AsyncSession,
        source_lang: str | None = None,
        target_lang: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GlossaryDB]:
        stmt = select(GlossaryDB)
        if source_lang:
            stmt = stmt.where(GlossaryDB.source_lang == source_lang)
        if target_lang:
            stmt = stmt.where(GlossaryDB.target_lang == target_lang)
        stmt = stmt.limit(limit).offset(offset)
        return list((await session.execute(stmt)).scalars().all())

    async def count(self, session: AsyncSession) -> int:
        return (
            await session.execute(select(func.count(GlossaryDB.id)))
        ).scalar_one()

    async def delete(self, session: AsyncSession, entry_id: str) -> bool:
        result = await session.execute(
            delete(GlossaryDB).where(GlossaryDB.id == entry_id)
        )
        await session.commit()
        self._query_cache.clear()
        return result.rowcount > 0

    async def find_relevant(
        self, session: AsyncSession, source_text: str, source_lang: str, target_lang: str
    ) -> list[dict[str, Any]]:
        """Термины, встречающиеся в исходном тексте (с кэшем запросов)."""
        key = (source_lang, target_lang)
        if key not in self._query_cache:
            rows = await self.list(session, source_lang, target_lang, limit=10000)
            self._query_cache[key] = [
                {"source_text": r.source_text, "target_text": r.target_text}
                for r in rows
            ]
        lowered = source_text.lower()
        return [
            e
            for e in self._query_cache[key]
            if e["source_text"].lower() in lowered
        ]


glossary_store = GlossaryStore()
