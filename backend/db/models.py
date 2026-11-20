import hashlib
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def deterministic_id(source_lang: str, target_lang: str, text: str) -> str:
    """Детерминированный id: <src>|<tgt>|<sha256>.

    Позволяет избежать UniqueConstraint на текстовых полях (лимит B-tree
    в Postgres — 2704 байта) и делает импорт идемпотентным через
    INSERT ... ON CONFLICT (id) DO NOTHING.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{source_lang}|{target_lang}|{digest}"


class GlossaryDB(Base):
    """Справочник: канцеляризм -> простой эквивалент."""

    __tablename__ = "glossaries"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    target_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_glossaries_langs", "source_lang", "target_lang"),)


class TMDB(Base):
    """База знаний: сложное предложение -> простое предложение."""

    __tablename__ = "translation_memories"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    target_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(32), nullable=False, default="general")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_tm_langs", "source_lang", "target_lang"),)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UncertainExample(Base):
    """Очередь HITL: неоднозначные результаты на ручную проверку."""

    __tablename__ = "uncertain_examples"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    target_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    machine_translation: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_translation: Mapped[str | None] = mapped_column(Text, nullable=True)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_passed: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # pending / confirmed / rejected
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_uncertain_status_created", "status", "created_at"),
        Index(
            "ix_uncertain_langs_status_created",
            "source_lang",
            "target_lang",
            "status",
            "created_at",
        ),
    )


# NULL в UNIQUE-индексе не гарантирует уникальность — используем сентинел.
CACHE_SENTINEL = "unknown"


class TranslationCache(Base):
    __tablename__ = "translation_cache"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    target_lang: Mapped[str] = mapped_column(String(16), nullable=False)
    engine_used: Mapped[str] = mapped_column(
        String(32), nullable=False, default=CACHE_SENTINEL
    )
    eval_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default=CACHE_SENTINEL
    )
    model_used: Mapped[str] = mapped_column(
        String(128), nullable=False, default=CACHE_SENTINEL
    )
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    translated_text: Mapped[str] = mapped_column(Text, nullable=False)
    hits: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "source_lang",
            "target_lang",
            "engine_used",
            "eval_mode",
            "model_used",
            "source_text",
            name="uq_translation_cache_key",
        ),
        Index("ix_translation_cache_eval_mode", "eval_mode"),
    )
