"""Основной пайплайн упрощения текста + 8 ablation-режимов."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.critic import CriticAgent
from backend.agents.editor import EditorAgent
from backend.agents.supervisor import SupervisorAgent
from backend.agents.validator import ValidatorAgent
from backend.core.config import get_settings
from backend.core.exceptions import UnknownEvalModeError
from backend.data.cache import cache_store
from backend.data.glossary import glossary_store
from backend.data.translator_manager import translator_manager
from backend.utils.logging import logger

# Режимы: (use_glossary, tm_mode, use_critic, use_editor)
EVAL_MODES: dict[str, dict[str, Any]] = {
    "baseline": {"glossary": False, "tm": None, "critic": False, "editor": False},
    "glossary": {"glossary": True, "tm": None, "critic": False, "editor": False},
    "tm": {"glossary": True, "tm": "bm25", "critic": False, "editor": False},
    "tm_hybrid": {"glossary": True, "tm": "hybrid", "critic": False, "editor": False},
    "agents_only": {"glossary": False, "tm": None, "critic": True, "editor": True},
    "tm_hybrid_critic": {"glossary": True, "tm": "hybrid", "critic": True, "editor": False},
    "tm_hybrid_editor": {"glossary": True, "tm": "hybrid", "critic": False, "editor": True},
    "full": {"glossary": True, "tm": "hybrid", "critic": True, "editor": True},
}

_MAX_TM_EXAMPLES = 5


def get_mode_config(eval_mode: str) -> dict[str, Any]:
    if eval_mode not in EVAL_MODES:
        raise UnknownEvalModeError(
            f"Неизвестный режим {eval_mode!r}", context={"modes": sorted(EVAL_MODES)}
        )
    return EVAL_MODES[eval_mode]


def build_prompt(
    source_text: str,
    glossary: list[dict[str, str]],
    tm_examples: list[dict[str, Any]],
) -> str:
    parts = [
        "Ты — эксперт по упрощению официальных текстов. Перепиши канцелярский текст "
        "простым русским языком, сохранив все факты, даты, суммы и имена.",
    ]
    if glossary:
        lines = "\n".join(
            f"- «{g['source_text']}» → «{g['target_text']}»" for g in glossary[:20]
        )
        parts.append(f"Замени канцеляризмы:\n{lines}")
    if tm_examples:
        lines = "\n".join(
            f"- «{r['entry']['source_text']}» → «{r['entry']['target_text']}» "
            f"(score: {r['score']:.2f})"
            for r in tm_examples[:_MAX_TM_EXAMPLES]
        )
        parts.append(f"Примеры упрощения (чем выше score, тем надёжнее):\n{lines}")
    parts.append(
        f"\nКанцелярский текст:\n{source_text}\n\n"
        "Верни ТОЛЬКО упрощённый текст, без пояснений.\n\nПростой текст:"
    )
    return "\n\n".join(parts)


async def _retrieve_tm(
    mode: str, session: AsyncSession, source_text: str, source_lang: str, target_lang: str
) -> list[dict[str, Any]]:
    from backend.retrieval.bm25_index import BM25Index

    if mode == "bm25":
        from backend.services.pipeline_state import get_bm25_index

        return get_bm25_index().search(source_text, source_lang, target_lang)
    # hybrid
    from backend.services.pipeline_state import get_hybrid_retriever

    retriever = await get_hybrid_retriever(session)
    return await retriever.search(source_text, source_lang, target_lang)


async def perform_translation(
    session: AsyncSession,
    source_text: str,
    source_lang: str = "bureau",
    target_lang: str = "plain",
    domain: str = "general",
    eval_mode: str | None = None,
    engine: str | None = None,
    strict_engine: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    mode = eval_mode or settings.EVAL_MODE
    cfg = get_mode_config(mode)

    # Кэш (только для cacheable-домена)
    cacheable = domain == settings.CACHEABLE_DOMAIN and use_cache
    if cacheable:
        cached = await cache_store.get(
            session, source_lang, target_lang, source_text, engine=engine, eval_mode=mode
        )
        if cached is not None:
            logger.info("Кэш-хит для текста длиной %d", len(source_text))
            return {
                "source_text": source_text,
                "translated_text": cached.translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": domain,
                "eval_mode": mode,
                "engine_used": cached.engine_used,
                "model_used": cached.model_used,
                "from_cache": True,
            }

    # Retrieval
    glossary: list[dict[str, str]] = []
    if cfg["glossary"]:
        glossary = await glossary_store.find_relevant(
            session, source_text, source_lang, target_lang
        )
    tm_examples: list[dict[str, Any]] = []
    if cfg["tm"]:
        tm_examples = await _retrieve_tm(cfg["tm"], session, source_text, source_lang, target_lang)

    # LLM
    prompt = build_prompt(source_text, glossary, tm_examples)
    translation = await translator_manager.translate(
        prompt, source_lang, target_lang, engine=engine, strict_engine=strict_engine
    )

    state: dict[str, Any] = {
        "source_text": source_text,
        "translated_text": translation["text"],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "domain": domain,
        "glossary_used": glossary,
        "tm_examples": [r["entry"] for r in tm_examples],
    }

    # Агенты
    supervisor = SupervisorAgent(
        ValidatorAgent(),
        CriticAgent() if cfg["critic"] else None,
        EditorAgent() if cfg["editor"] else None,
        use_critic=cfg["critic"],
        use_editor=cfg["editor"],
    )
    state = await supervisor.process(state)

    result = {
        **{k: state.get(k) for k in (
            "source_text", "translated_text", "source_lang", "target_lang", "domain",
            "validation_passed", "validator_errors", "validator_warnings",
            "critic_score", "critic_passed", "critic_failed", "critic_suggestions",
            "editor_changed", "editor_rolled_back", "stopped_reason",
            "final_success", "reasoning_log",
        )},
        "eval_mode": mode,
        "engine_used": translation["engine_used"],
        "model_used": translation["model_used"],
        "from_cache": False,
    }

    # HITL: неоднозначные результаты — в очередь
    if not state.get("final_success"):
        from backend.hitl.active_learning import push_uncertain

        await push_uncertain(session, state)

    # Кэш только успешных результатов cacheable-домена
    if cacheable and state.get("final_success"):
        await cache_store.put(
            session,
            source_lang,
            target_lang,
            source_text,
            state["translated_text"],
            engine=translation["engine_used"],
            model=translation["model_used"],
            eval_mode=mode,
        )

    return result
