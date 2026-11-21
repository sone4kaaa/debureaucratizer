"""Юнит-тесты HITL: очередь неоднозначных примеров + feedback в TM."""

import pytest

from backend.core.exceptions import UncertainExampleNotFoundError
from backend.data.translation_memory import tm_store
from backend.hitl import active_learning, feedback


def _state(text: str = "Сложный канцелярский текст") -> dict:
    return {
        "source_lang": "bureau",
        "target_lang": "plain",
        "source_text": text,
        "translated_text": "Машинный простой текст",
        "critic_score": 4.5,
        "validation_passed": False,
    }


async def test_push_and_get_pending(test_db) -> None:
    async with test_db() as session:
        example = await active_learning.push_uncertain(session, _state())
        assert example.status == "pending"
        assert example.critic_score == 4.5

        pending = await active_learning.get_pending(session)
        assert len(pending) == 1
        # фильтры по паре стилей
        assert await active_learning.get_pending(session, source_lang="legal") == []
        assert len(await active_learning.get_pending(session, source_lang="bureau")) == 1


async def test_confirm_with_correction_goes_to_tm(test_db) -> None:
    async with test_db() as session:
        example = await active_learning.push_uncertain(session, _state())
        confirmed = await active_learning.confirm(
            session, example.id, corrected="Просто и понятно"
        )
        assert confirmed.status == "confirmed"
        assert confirmed.confirmed_at is not None
        assert confirmed.corrected_translation == "Просто и понятно"

        tm_rows = await tm_store.get_all(session, "bureau", "plain")
        assert len(tm_rows) == 1
        assert tm_rows[0].target_text == "Просто и понятно"


async def test_confirm_without_correction_uses_machine_translation(test_db) -> None:
    async with test_db() as session:
        example = await active_learning.push_uncertain(session, _state())
        await active_learning.confirm(session, example.id)
        tm_rows = await tm_store.get_all(session, "bureau", "plain")
        assert tm_rows[0].target_text == "Машинный простой текст"


async def test_confirm_unknown_id_404(test_db) -> None:
    async with test_db() as session:
        with pytest.raises(UncertainExampleNotFoundError):
            await active_learning.confirm(session, "не-существующий-id")
        with pytest.raises(UncertainExampleNotFoundError):
            await active_learning.reject(session, "не-существующий-id")


async def test_reject_and_stats(test_db) -> None:
    async with test_db() as session:
        e1 = await active_learning.push_uncertain(session, _state("Текст один"))
        e2 = await active_learning.push_uncertain(session, _state("Текст два"))
        await active_learning.confirm(session, e1.id)
        rejected = await active_learning.reject(session, e2.id)
        assert rejected.status == "rejected"

        assert await active_learning.get_pending(session) == []
        stats = await active_learning.stats(session)
        assert stats == {"pending": 0, "confirmed": 1, "rejected": 1}


async def test_feedback_save_correction(test_db) -> None:
    async with test_db() as session:
        added = await feedback.save_correction(
            session, "bureau", "plain", "Осуществить оплату", "Оплатить"
        )
        assert added == 1
        # повтор — идемпотентно
        assert (
            await feedback.save_correction(
                session, "bureau", "plain", "Осуществить оплату", "Оплатить"
            )
            == 0
        )
