import asyncio

from celery import Celery
from celery.signals import worker_shutdown

from backend.core.config import get_settings
from backend.utils.logging import logger

settings = get_settings()

celery_app = Celery(
    "debureaucratizer",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["backend.worker.tasks"],
)
celery_app.conf.update(task_track_started=True, result_expires=3600)

# Единый event loop на процесс worker'а — иначе async-объекты
# (engine, httpx-клиенты) оказываются «attached to a different loop».
_worker_loop: asyncio.AbstractEventLoop | None = None


def get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


@worker_shutdown.connect
def _on_worker_shutdown(**kwargs) -> None:
    global _worker_loop
    if _worker_loop is not None and not _worker_loop.is_closed():
        from backend.utils.http_client import close_all_clients

        _worker_loop.run_until_complete(close_all_clients())
        _worker_loop.close()
        _worker_loop = None
        logger.info("Worker: event loop закрыт")
