"""Общие хелперы импорта: парсинг CSV/JSON, загрузка builtin_data.

CSV-форматы (оба поддерживаются, различие только в заголовке):
  - legacy:     source_lang,target_lang,en,tgt
  - lang-coded: source_lang,target_lang,<src_code>,<tgt_code>  (напр. ar,ru)
JSON (для TM): {"source_lang": "...", "target_lang": "...",
               "entries": [{"en": "...", "tgt": "..."}]}
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_PATH = Path(__file__).resolve().parent / "builtin_data.py"


def load_builtin():
    """Загружает builtin_data.py по пути (пакет scripts.import не импортируем:
    import — ключевое слово)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("builtin_data", BUILTIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_content(content: bytes) -> list[dict[str, str]]:
    """CSV -> [{source_lang, target_lang, source_text, target_text}].

    Колонки 3-4 — тексты независимо от имён заголовков (en/tgt или коды языков).
    """
    reader = csv.reader(io.StringIO(content.decode("utf-8")))
    rows = [row for row in reader if row]
    if len(rows) < 2:
        return []
    entries = []
    for row in rows[1:]:
        if len(row) < 4 or not row[2].strip() or not row[3].strip():
            continue
        entries.append(
            {
                "source_lang": row[0].strip() or "bureau",
                "target_lang": row[1].strip() or "plain",
                "source_text": row[2].strip(),
                "target_text": row[3].strip(),
            }
        )
    return entries


def parse_json_content(content: bytes) -> list[dict[str, str]]:
    """JSON -> [{source_lang, target_lang, source_text, target_text}]."""
    data = json.loads(content.decode("utf-8"))
    source_lang = data.get("source_lang", "bureau")
    target_lang = data.get("target_lang", "plain")
    entries = []
    for item in data.get("entries", []):
        src = (item.get("en") or item.get("source_text") or "").strip()
        tgt = (item.get("tgt") or item.get("target_text") or "").strip()
        if src and tgt:
            entries.append(
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_text": src,
                    "target_text": tgt,
                }
            )
    return entries


def load_entries(path: Path) -> list[dict[str, str]]:
    content = path.read_bytes()
    if path.suffix.lower() == ".json":
        return parse_json_content(content)
    return parse_csv_content(content)


def dedup(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Дедупликация внутри файла по (source_lang, target_lang, source_text)."""
    seen: set[tuple[str, str, str]] = set()
    result = []
    for e in entries:
        key = (e["source_lang"], e["target_lang"], e["source_text"])
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result
