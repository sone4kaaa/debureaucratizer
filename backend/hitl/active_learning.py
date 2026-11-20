"""Очередь неоднозначных примеров для ручной проверки (Human-in-the-Loop)."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import UncertainExampleNotFoundError
from backend.db.models import UncertainExample


async def push_uncertain(session: AsyncSession, state: dict[str, Any]) -> UncertainExample:
    example = UncertainExample(
        source_lang=state.get("source_lang", "bureau"),
        target_lang=state.get("target_lang", "plain"),
        source_text=state.get("source_text", ""),
        machine_translation=state.get("translated_text", ""),
        critic_score=state.get("critic_score"),
        validation_passed=bool(state.get("validation_passed")),
        status="pending",
    )
    session.add(example)
    await session.commit()
    await session.refresh(example)
    return example


async def get_pending(
    session: AsyncSession,
    source_lang: str | None = None,
    target_lang: str | None = None,
    limit: int = 50,
) -> list[UncertainExample]:
    stmt = (
        select(UncertainExample)
        .where(UncertainExample.status == "pending")
        .order_by(UncertainExample.created_at)
        .limit(limit)
    )
    if source_lang:
        stmt = stmt.where(UncertainExample.source_lang == source_lang)
    if target_lang:
        stmt = stmt.where(UncertainExample.target_lang == target_lang)
    return list((await session.execute(stmt)).scalars().all())


async def _get_or_404(session: AsyncSession, example_id: str) -> UncertainExample:
    example = (
        await session.execute(
            select(UncertainExample).where(UncertainExample.id == example_id)
        )
    ).scalar_one_or_none()
    if example is None:
        raise UncertainExampleNotFoundError(f"Пример {example_id} не найден")
    return example


async def confirm(
    session: AsyncSession, example_id: str, corrected: str | None = None
) -> UncertainExample:
    example = await _get_or_404(session, example_id)
    example.status = "confirmed"
    example.confirmed_at = datetime.now(UTC)
    if corrected:
        example.corrected_translation = corrected
    await session.commit()
    await session.refresh(example)

    # Подтверждённая правка пополняет TM
    from backend.data.translation_memory import tm_store

    await tm_store.insert_in_session(
        session,
        [
            {
                "source_text": example.source_text,
                "target_text": corrected or example.machine_translation,
            }
        ],
        example.source_lang,
        example.target_lang,
    )
    return example


async def reject(session: AsyncSession, example_id: str) -> UncertainExample:
    example = await _get_or_404(session, example_id)
    example.status = "rejected"
    example.confirmed_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(example)
    return example


async def stats(session: AsyncSession) -> dict[str, int]:
    rows = (
        await session.execute(
            select(UncertainExample.status, func.count()).group_by(UncertainExample.status)
        )
    ).all()
    result = {"pending": 0, "confirmed": 0, "rejected": 0}
    for status, count in rows:
        result[status] = count
    return result
