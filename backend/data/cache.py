"""Кэш результатов упрощения в БД.

Ключ: (source_lang, target_lang, engine_used, eval_mode, model_used, source_text).
NULL-поля хранятся как сентинел "unknown" (NULL в UNIQUE-индексе не гарантирует
уникальность). На чтении None = «любое значение».
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import CACHE_SENTINEL, TranslationCache


def _norm(value: str | None) -> str:
    return value if value else CACHE_SENTINEL


class CacheStore:
    async def get(
        self,
        session: AsyncSession,
        source_lang: str,
        target_lang: str,
        source_text: str,
        engine: str | None = None,
        model: str | None = None,
        eval_mode: str | None = None,
    ) -> TranslationCache | None:
        stmt = select(TranslationCache).where(
            TranslationCache.source_lang == source_lang,
            TranslationCache.target_lang == target_lang,
            TranslationCache.source_text == source_text,
        )
        if engine is not None:
            stmt = stmt.where(TranslationCache.engine_used == _norm(engine))
        if model is not None:
            stmt = stmt.where(TranslationCache.model_used == _norm(model))
        if eval_mode is not None:
            stmt = stmt.where(TranslationCache.eval_mode == _norm(eval_mode))
        row = (await session.execute(stmt)).scalars().first()
        if row is not None:
            row.hits += 1
            await session.commit()
        return row

    async def put(
        self,
        session: AsyncSession,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translated_text: str,
        engine: str | None = None,
        model: str | None = None,
        eval_mode: str | None = None,
    ) -> TranslationCache:
        row = TranslationCache(
            source_lang=source_lang,
            target_lang=target_lang,
            engine_used=_norm(engine),
            eval_mode=_norm(eval_mode),
            model_used=_norm(model),
            source_text=source_text,
            translated_text=translated_text,
        )
        session.add(row)
        try:
            await session.commit()
        except Exception:
            await session.rollback()  # дубликат ключа — не страшно
        return row

    async def clear(
        self,
        session: AsyncSession,
        older_than_days: int | None = None,
        eval_mode: str | None = None,
    ) -> int:
        stmt = delete(TranslationCache)
        if older_than_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
            stmt = stmt.where(TranslationCache.created_at < cutoff)
        if eval_mode is not None:
            stmt = stmt.where(TranslationCache.eval_mode == _norm(eval_mode))
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0

    async def stats(self, session: AsyncSession) -> dict[str, Any]:
        total = (
            await session.execute(select(func.count(TranslationCache.id)))
        ).scalar_one()
        hits = (
            await session.execute(select(func.sum(TranslationCache.hits)))
        ).scalar_one() or 0
        return {"total_entries": total, "total_hits": hits}


cache_store = CacheStore()
