from fastapi import APIRouter, Response
from sqlalchemy import text

from backend.core.config import get_settings
from backend.db.session import engine
from backend.utils.http_client import get_http_client

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def _check_db() -> str:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return "healthy"


async def _check_redis() -> str:
    import redis.asyncio as aioredis

    settings = get_settings()
    client = aioredis.from_url(settings.redis_url, socket_timeout=2)
    try:
        await client.ping()
        return "healthy"
    finally:
        await client.aclose()


async def _check_qdrant() -> str:
    settings = get_settings()
    client = get_http_client(timeout=3.0)
    resp = await client.get(f"{settings.qdrant_url}/collections", timeout=3.0)
    resp.raise_for_status()
    return "healthy"


async def _check_ollama() -> str:
    settings = get_settings()
    client = get_http_client(timeout=3.0)
    resp = await client.get(f"{settings.ollama_url}/api/tags", timeout=3.0)
    resp.raise_for_status()
    return "healthy"


@router.get("/health")
async def health(response: Response) -> dict[str, object]:
    checks: dict[str, str] = {}
    degraded = False
    for name, fn in (
        ("database", _check_db),
        ("redis", _check_redis),
        ("qdrant", _check_qdrant),
        ("ollama", _check_ollama),
    ):
        try:
            checks[name] = await fn()
        except Exception:
            checks[name] = "unhealthy"
            degraded = True
    if degraded:
        response.status_code = 503
    return {
        "status": "degraded" if degraded else "ok",
        "models": get_settings().ensemble_models_list,
        **checks,
    }
