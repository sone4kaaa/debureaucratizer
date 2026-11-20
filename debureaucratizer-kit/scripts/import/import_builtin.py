"""Импорт встроенных эталонных данных в БД + утилиты администрирования.

Загружает в базу:
  - глоссарий «канцеляризм -> простой эквивалент» (GLOSSARY_PAIRS, 651 пара);
  - память переводов «сложное -> простое» (generate_tm_pairs(), ~5900 пар).

Вставка идемпотентна: повторный запуск не создаёт дубликатов
(deterministic id + ON CONFLICT DO NOTHING / проверка существования).

Использование (из корня проекта):
  python -m poetry run python scripts/import/import_builtin.py

  # назначить пользователя админом (email или username):
  python -m poetry run python scripts/import/import_builtin.py --admin tester

  # только показать, сколько данных будет импортировано:
  python -m poetry run python scripts/import/import_builtin.py --dry-run

После импорта перезапустите API: TM подхватывается в BM25/Qdrant при старте
(lifespan -> ensure_indexes_loaded). Если API уже запущен — просто
перезапустите uvicorn, повторный импорт не требуется.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# builtin_data лежит рядом, но пакет «scripts.import» не импортируется
# напрямую (import — ключевое слово языка), поэтому грузим по пути файла.
_BUILTIN_PATH = Path(__file__).resolve().parent / "builtin_data.py"
_spec = importlib.util.spec_from_file_location("builtin_data", _BUILTIN_PATH)
builtin_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builtin_data)

TM_CHUNK_SIZE = 500  # строк в одном INSERT (запас по лимиту bind-параметров SQLite)


async def _create_schema() -> None:
    from backend.db.models import Base
    from backend.db.session import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def import_glossary(dry_run: bool = False) -> int:
    from backend.data.glossary import glossary_store
    from backend.db.session import AsyncSessionLocal

    # GLOSSARY_PAIRS содержит внутренние дубликаты по source_text,
    # а glossary_store.add_bulk проверяет существование только в БД,
    # поэтому дедуплицируем сами (иначе UNIQUE constraint на пакете).
    seen: set[tuple[str, str, str]] = set()
    pairs = []
    for s, t in builtin_data.GLOSSARY_PAIRS:
        key = ("bureau", "plain", s)
        if key not in seen:
            seen.add(key)
            pairs.append((s, t))
    if dry_run:
        return len(pairs)
    entries = [
        {"source_lang": "bureau", "target_lang": "plain", "source_text": s, "target_text": t}
        for s, t in pairs
    ]
    async with AsyncSessionLocal() as session:
        added = await glossary_store.add_bulk(session, entries)
    return added


async def import_tm(dry_run: bool = False) -> int:
    from backend.data.translation_memory import tm_store
    from backend.db.session import AsyncSessionLocal

    pairs = builtin_data.generate_tm_pairs()
    if dry_run:
        return len(pairs)
    added = 0
    async with AsyncSessionLocal() as session:
        for i in range(0, len(pairs), TM_CHUNK_SIZE):
            chunk = [
                {"source_text": s, "target_text": t} for s, t in pairs[i : i + TM_CHUNK_SIZE]
            ]
            added += await tm_store.insert_in_session(session, chunk, "bureau", "plain")
    return added


async def make_admin(login: str) -> bool:
    from sqlalchemy import select, update

    from backend.db.models import User
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        user = (
            await session.execute(
                select(User).where((User.email == login) | (User.username == login))
            )
        ).scalar_one_or_none()
        if user is None:
            return False
        if user.role == "admin":
            print(f"[admin] {login} уже admin")
            return True
        await session.execute(update(User).where(User.id == user.id).values(role="admin"))
        await session.commit()
        print(f"[admin] {login} теперь admin")
        return True


async def show_counts() -> None:
    from sqlalchemy import func, select

    from backend.db.models import GlossaryDB, TMDB
    from backend.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        g = (await session.execute(select(func.count(GlossaryDB.id)))).scalar_one()
        t = (await session.execute(select(func.count(TMDB.id)))).scalar_one()
    print(f"В БД сейчас: glossaries={g}, translation_memories={t}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--admin", metavar="LOGIN", help="назначить админом (email или username)")
    parser.add_argument("--dry-run", action="store_true", help="показать объём без записи в БД")
    parser.add_argument("--stats", action="store_true", help="показать текущее число записей в БД")
    args = parser.parse_args()

    if args.stats:
        await show_counts()
        return

    if args.admin:
        ok = await make_admin(args.admin)
        if not ok:
            print(f"[admin] пользователь {args.admin!r} не найден")
            sys.exit(1)
        return

    print(f"Найдено в builtin_data: glossary={len(builtin_data.GLOSSARY_PAIRS)} пар")
    tm_count = len(builtin_data.generate_tm_pairs())
    print(f"                        tm={tm_count} пар")

    await _create_schema()
    g_added = await import_glossary(dry_run=args.dry_run)
    t_added = await import_tm(dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] было бы добавлено: glossary={g_added}, tm={t_added}")
        return
    print(f"Импортировано: glossary +{g_added}, tm +{t_added}")
    await show_counts()
    print("Готово. Если API запущен — перезапустите его, чтобы TM попала в BM25/Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())
