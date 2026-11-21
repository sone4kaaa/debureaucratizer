"""Импорт памяти переводов «сложное -> простое» (TM) в БД.

Использование (из корня проекта):
  python scripts/import/add_tm.py data/import/tm.csv
  python scripts/import/add_tm.py data/import/tm.json
  python scripts/import/add_tm.py --builtin      # ~5900 пар из шаблонов
  python scripts/import/add_tm.py --list

Идемпотентно: INSERT ... ON CONFLICT (id) DO NOTHING (Postgres и SQLite).
Лог: added / skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_common():
    """common.py грузим по пути: «scripts.import» в import-выражении
    недопустим (import — ключевое слово)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "import_common", Path(__file__).resolve().parent / "common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_common = _load_common()
dedup = _common.dedup
load_builtin = _common.load_builtin
load_entries = _common.load_entries

CHUNK = 500  # строк в одном INSERT (запас по лимиту bind-параметров)


def builtin_entries() -> list[dict[str, str]]:
    data = load_builtin()
    return [
        {"source_lang": "bureau", "target_lang": "plain", "source_text": s, "target_text": t}
        for s, t in data.generate_tm_pairs()
    ]


async def import_entries(entries: list[dict[str, str]]) -> tuple[int, int]:
    from backend.data.translation_memory import tm_store
    from backend.db.models import Base
    from backend.db.session import AsyncSessionLocal, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    added = 0
    async with AsyncSessionLocal() as session:
        by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
        for e in entries:
            by_pair.setdefault((e["source_lang"], e["target_lang"]), []).append(e)
        for (src, tgt), group in by_pair.items():
            for i in range(0, len(group), CHUNK):
                added += await tm_store.insert_in_session(
                    session, group[i : i + CHUNK], src, tgt
                )
    return added, len(entries) - added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", help="путь к CSV/JSON с TM")
    parser.add_argument("--builtin", action="store_true", help="загрузить встроенный набор")
    parser.add_argument("--list", action="store_true", help="показать встроенные наборы")
    args = parser.parse_args()

    if args.list:
        entries = builtin_entries()
        print(f"builtin TM: {len(entries)} пар (bureau -> plain)")
        for e in entries[:5]:
            print(f"  «{e['source_text']}» -> «{e['target_text']}»")
        print("  ...")
        return

    if args.builtin:
        entries = builtin_entries()
    elif args.file:
        entries = dedup(load_entries(Path(args.file)))
    else:
        parser.error("укажите файл, --builtin или --list")

    added, skipped = asyncio.run(import_entries(entries))
    print(f"TM: added={added}, skipped={skipped}")
    print("Перезапустите API (или вызовите задачу index_translation_memory),")
    print("чтобы новые пары попали в BM25/Qdrant.")


if __name__ == "__main__":
    main()
