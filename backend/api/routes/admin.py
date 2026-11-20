from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db, require_admin
from backend.data.cache import cache_store
from backend.data.glossary import glossary_store
from backend.data.translation_memory import tm_store
from backend.db.models import User

router = APIRouter(prefix="/admin", tags=["admin"])


class EntryIn(BaseModel):
    source_lang: str = "bureau"
    target_lang: str = "plain"
    source_text: str
    target_text: str


def _parse_csv(content: bytes) -> list[dict[str, str]]:
    """CSV-форматы: legacy `source_lang,target_lang,en,tgt` или lang-coded
    `source_lang,target_lang,<src_code>,<tgt_code>`."""
    import csv
    import io

    reader = csv.reader(io.StringIO(content.decode("utf-8")))
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    entries = []
    for row in rows[1:]:
        if len(row) < 4:
            continue
        entries.append(
            {
                "source_lang": row[0].strip(),
                "target_lang": row[1].strip(),
                "source_text": row[2].strip(),
                "target_text": row[3].strip(),
            }
        )
    return entries


# --- Glossary ---


@router.post("/glossary/add")
async def glossary_add(
    payload: EntryIn,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, Any]:
    entry = await glossary_store.add(
        db, payload.source_lang, payload.target_lang, payload.source_text, payload.target_text
    )
    return {"id": entry.id}


@router.post("/glossary/import-csv")
async def glossary_import(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, int]:
    entries = _parse_csv(await file.read())
    added = await glossary_store.add_bulk(db, entries)
    return {"added": added, "total": len(entries)}


@router.get("/glossary/list")
async def glossary_list(
    source_lang: str | None = None,
    target_lang: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, Any]:
    rows = await glossary_store.list(db, source_lang, target_lang, limit, offset)
    return {
        "total": await glossary_store.count(db),
        "items": [
            {"id": r.id, "source_text": r.source_text, "target_text": r.target_text}
            for r in rows
        ],
    }


@router.delete("/glossary/delete/{entry_id}")
async def glossary_delete(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, bool]:
    if not await glossary_store.delete(db, entry_id):
        raise HTTPException(status_code=404, detail="Запись не найдена")
    return {"deleted": True}


# --- TM ---


@router.post("/tm/add")
async def tm_add(
    payload: EntryIn,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, int]:
    added = await tm_store.insert_in_session(
        db,
        [{"source_text": payload.source_text, "target_text": payload.target_text}],
        payload.source_lang,
        payload.target_lang,
    )
    return {"added": added}


@router.post("/tm/import-csv")
async def tm_import(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, int]:
    entries = _parse_csv(await file.read())
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for e in entries:
        grouped.setdefault((e["source_lang"], e["target_lang"]), []).append(e)
    added = 0
    for (src, tgt), group in grouped.items():
        added += await tm_store.insert_in_session(db, group, src, tgt)
    return {"added": added, "total": len(entries)}


@router.get("/tm/list")
async def tm_list(
    source_lang: str | None = None,
    target_lang: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, Any]:
    rows = await tm_store.list(db, source_lang, target_lang, limit, offset)
    return {
        "total": await tm_store.count(db),
        "items": [
            {"id": r.id, "source_text": r.source_text, "target_text": r.target_text}
            for r in rows
        ],
    }


@router.delete("/tm/delete/{entry_id}")
async def tm_delete(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, bool]:
    if not await tm_store.delete(db, entry_id):
        raise HTTPException(status_code=404, detail="Запись не найдена")
    return {"deleted": True}


# --- Cache ---


@router.delete("/cache/clear")
async def cache_clear(
    eval_mode: str | None = None,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, int]:
    deleted = await cache_store.clear(db, eval_mode=eval_mode)
    return {"deleted": deleted}


@router.get("/cache/stats")
async def cache_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, Any]:
    return await cache_store.stats(db)
