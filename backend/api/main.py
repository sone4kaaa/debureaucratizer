from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.api.routes import admin, auth, eval, health, hitl, translate
from backend.core.config import get_settings
from backend.core.constants import STYLES
from backend.core.exceptions import RUNGException
from backend.db.models import Base
from backend.db.session import engine
from backend.utils.http_client import close_all_clients
from backend.utils.logging import logger

settings = get_settings()

# Идемпотентные миграции индексов для уже существующих БД
_POST_CREATE_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS ix_glossaries_langs ON glossaries (source_lang, target_lang)",
    "CREATE INDEX IF NOT EXISTS ix_tm_langs ON translation_memories (source_lang, target_lang)",
    "CREATE INDEX IF NOT EXISTS ix_uncertain_status_created ON uncertain_examples (status, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_uncertain_langs_status_created ON uncertain_examples (source_lang, target_lang, status, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_translation_cache_eval_mode ON translation_cache (eval_mode)",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Запуск %s (DEBUG=%s)", settings.PROJECT_NAME, settings.DEBUG)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for stmt in _POST_CREATE_STATEMENTS:
            await conn.execute(text(stmt))
    logger.info("Схема БД готова")

    from backend.db.session import AsyncSessionLocal
    from backend.services.pipeline_state import ensure_indexes_loaded

    async with AsyncSessionLocal() as session:
        try:
            await ensure_indexes_loaded(session)
        except Exception as exc:
            logger.warning("Индексы TM не загружены при старте: %s", exc)

    yield
    await close_all_clients()
    await engine.dispose()
    logger.info("Остановка приложения")


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RUNGException)
async def rung_exception_handler(request: Request, exc: RUNGException) -> JSONResponse:
    logger.error("RUNGException: %s (%s)", exc.message, exc.code)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


for r in (health.router, auth.router, translate.router, admin.router, eval.router, hitl.router):
    app.include_router(r)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": settings.PROJECT_NAME}


@app.get("/languages")
async def languages() -> dict[str, dict[str, str]]:
    return {"styles": STYLES}
