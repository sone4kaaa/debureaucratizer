import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.core.constants import KNOWN_STYLES
from backend.core.exceptions import RUNGException
from backend.db.models import User
from backend.services.pipeline import perform_translation

router = APIRouter(prefix="/translate", tags=["translate"])

_SEMAPHORE = asyncio.Semaphore(4)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    source_lang: str = "bureau"
    target_lang: str = "plain"
    domain: str = "general"
    eval_mode: str | None = None
    engine: str | None = None


class MultiTranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    target_langs: list[str] = ["plain"]
    source_lang: str = "bureau"
    domain: str = "general"


class BatchTranslateRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    source_lang: str = "bureau"
    target_lang: str = "plain"
    domain: str = "general"


class EnsembleRequest(BaseModel):
    text: str = Field(min_length=1)
    source_lang: str = "bureau"
    target_lang: str = "plain"
    domain: str = "general"
    models: list[str] | None = None


def _validate_pair(source_lang: str, target_lang: str) -> None:
    if source_lang not in KNOWN_STYLES or target_lang not in KNOWN_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемая пара: {source_lang} -> {target_lang}",
        )


@router.post("")
async def translate(
    payload: TranslateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return await perform_translation(
        db,
        payload.text,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        domain=payload.domain,
        eval_mode=payload.eval_mode,
        engine=payload.engine,
    )


@router.post("/multi")
async def translate_multi(
    payload: MultiTranslateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    async def _one(target: str) -> dict[str, Any]:
        async with _SEMAPHORE:
            return await perform_translation(
                db, payload.text, source_lang=payload.source_lang,
                target_lang=target, domain=payload.domain,
            )

    results = await asyncio.gather(*(_one(t) for t in payload.target_langs))
    return {"source_text": payload.text, "results": dict(zip(payload.target_langs, results))}


@router.post("/batch")
async def translate_batch(
    payload: BatchTranslateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    async def _one(text: str) -> dict[str, Any]:
        async with _SEMAPHORE:
            try:
                return await perform_translation(
                    db, text, source_lang=payload.source_lang,
                    target_lang=payload.target_lang, domain=payload.domain,
                )
            except RUNGException as exc:
                return {"source_text": text, "error": exc.to_dict()}

    results = await asyncio.gather(*(_one(t) for t in payload.texts))
    return {"count": len(results), "results": results}


@router.post("/ensemble")
async def translate_ensemble(
    payload: EnsembleRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    _validate_pair(payload.source_lang, payload.target_lang)

    from backend.agents.critic import CriticAgent
    from backend.core.config import get_settings
    from backend.services.pipeline import build_prompt

    settings = get_settings()
    models = payload.models or settings.ensemble_models_list
    prompt = build_prompt(payload.text, [], [])

    from backend.utils.llm_client import ollama_generate
    from backend.utils.llm_processing import clean_translation

    async def _one(model: str) -> dict[str, Any]:
        async with _SEMAPHORE:
            raw = await ollama_generate(prompt, model=model, temperature=0.3)
            return {"model": model, "text": clean_translation(raw)}

    variants = await asyncio.gather(*(_one(m) for m in models))

    critic = CriticAgent()
    best: dict[str, Any] | None = None
    for variant in variants:
        state = await critic.process(
            {
                "source_text": payload.text,
                "translated_text": variant["text"],
                "domain": payload.domain,
            }
        )
        variant["critic_score"] = state.get("critic_score")
        score = state.get("critic_score")
        if score is not None and (best is None or score > best["critic_score"]):
            best = variant

    return {
        "source_text": payload.text,
        "variants": variants,
        "best": best,
    }
