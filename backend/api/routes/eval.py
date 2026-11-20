from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.api.deps import get_current_user
from backend.db.models import User
from backend.eval.runner import evaluator

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalRequest(BaseModel):
    hypotheses: list[str] = Field(min_length=1)
    references: list[str] = Field(min_length=1)
    include_neural: bool = True


@router.post("/run")
async def run_eval(
    payload: EvalRequest, user: User = Depends(get_current_user)
) -> dict[str, Any]:
    return evaluator.run(
        payload.hypotheses, payload.references, include_neural=payload.include_neural
    )
