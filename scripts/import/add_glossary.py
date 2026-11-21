"""Импорт глоссария «канцеляризм -> простой эквивалент» в БД.

Использование (из корня проекта):
  # из CSV/JSON-файла:
  python scripts/import/add_glossary.py data/import/glossary.csv
  # builtin-набор (650 пар):
  python scripts/import/add_glossary.py --builtin
  # показать встроенные наборы:
  python scripts/import/add_glossary.py --list

Идемпотентно: deterministic id, повторный запуск не создаёт дублей.
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


def builtin_entries() -> list[dict[str, str]]:
    data = load_builtin()
    return dedup(
        [
            {
                "source_lang": "bureau",
                "target_lang": "plain",
                "source_text": s,
                "target_text": t,
            }
            for s, t in data.GLOSSARY_PAIRS
        ]
    )


async def import_entries(entries: list[dict[str, str]]) -> tuple[int, int]:
    from backend.data.glossary import glossary_store
    from backend.db.models import Base
    from backend.db.session import AsyncSessionLocal, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        added = await glossary_store.add_bulk(session, entries)
    return added, len(entries) - added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", help="путь к CSV/JSON с глоссарием")
    parser.add_argument("--builtin", action="store_true", help="загрузить встроенный набор")
    parser.add_argument("--list", action="store_true", help="показать встроенные наборы")
    args = parser.parse_args()

    if args.list:
        entries = builtin_entries()
        print(f"builtin glossary: {len(entries)} пар (bureau -> plain)")
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
    print(f"Глоссарий: added={added}, skipped={skipped}")


if __name__ == "__main__":
    main()
