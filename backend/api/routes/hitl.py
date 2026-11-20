from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db, require_editor
from backend.db.models import User
from backend.hitl import active_learning, feedback

router = APIRouter(prefix="/hitl", tags=["hitl"])


class CorrectionIn(BaseModel):
    source_lang: str = "bureau"
    target_lang: str = "plain"
    source_text: str
    corrected_text: str
    domain: str = "general"


class ConfirmIn(BaseModel):
    corrected_translation: str | None = None


@router.post("/save-correction")
async def save_correction(
    payload: CorrectionIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_editor),
) -> dict[str, int]:
    added = await feedback.save_correction(
        db,
        payload.source_lang,
        payload.target_lang,
        payload.source_text,
        payload.corrected_text,
        domain=payload.domain,
    )
    return {"added": added}


@router.get("/uncertain")
async def list_uncertain(
    source_lang: str | None = None,
    target_lang: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_editor),
) -> dict[str, Any]:
    rows = await active_learning.get_pending(db, source_lang, target_lang, limit)
    return {
        "items": [
            {
                "id": r.id,
                "source_text": r.source_text,
                "machine_translation": r.machine_translation,
                "critic_score": r.critic_score,
                "validation_passed": r.validation_passed,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


@router.post("/uncertain/{example_id}/confirm")
async def confirm_uncertain(
    example_id: str,
    payload: ConfirmIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_editor),
) -> dict[str, str]:
    example = await active_learning.confirm(db, example_id, payload.corrected_translation)
    return {"id": example.id, "status": example.status}


@router.post("/uncertain/{example_id}/reject")
async def reject_uncertain(
    example_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_editor),
) -> dict[str, str]:
    example = await active_learning.reject(db, example_id)
    return {"id": example.id, "status": example.status}


@router.get("/uncertain/stats")
async def uncertain_stats(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_editor),
) -> dict[str, int]:
    return await active_learning.stats(db)
