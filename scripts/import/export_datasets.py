"""Экспорт эталонных данных из builtin-наборов в data/import/ (CSV + JSON).

Артефакт сдачи: «набор эталонных данных (CSV/JSON) —
>= 500 терминов глоссария, >= 1000 TM-пар».

Использование (из корня проекта):
  python scripts/import/export_datasets.py [--out-dir data/import]
"""

from __future__ import annotations

import argparse
import csv
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "import"))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_builtin()

    # --- глоссарий: CSV (legacy-формат) ---
    glossary = dedup(
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
    glossary_csv = out / "glossary.csv"
    with glossary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_lang", "target_lang", "en", "tgt"])
        for e in glossary:
            writer.writerow([e["source_lang"], e["target_lang"], e["source_text"], e["target_text"]])

    # --- TM: CSV + JSON ---
    tm_pairs = data.generate_tm_pairs()
    tm_csv = out / "tm.csv"
    with tm_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_lang", "target_lang", "en", "tgt"])
        for s, t in tm_pairs:
            writer.writerow(["bureau", "plain", s, t])

    tm_json = out / "tm.json"
    tm_json.write_text(
        json.dumps(
            {
                "source_lang": "bureau",
                "target_lang": "plain",
                "entries": [{"en": s, "tgt": t} for s, t in tm_pairs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Записано: {glossary_csv.name} ({len(glossary)} терминов), "
          f"{tm_csv.name} + {tm_json.name} ({len(tm_pairs)} TM-пар)")


if __name__ == "__main__":
    main()
