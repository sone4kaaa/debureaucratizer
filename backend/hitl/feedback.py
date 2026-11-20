"""Ручные корректировки пользователей -> пополнение TM."""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.data.translation_memory import tm_store


async def save_correction(
    session: AsyncSession,
    source_lang: str,
    target_lang: str,
    source_text: str,
    corrected_text: str,
    domain: str = "general",
) -> int:
    return await tm_store.insert_in_session(
        session,
        [{"source_text": source_text, "target_text": corrected_text}],
        source_lang,
        target_lang,
        domain=domain,
    )
