"""Подготовка оценочных датасетов для прогона ablation-режимов.

Источники (в порядке приоритета):
  1. HuggingFace Datasets (--hf-dataset, HF_TOKEN из env) — если датасет
     «канцелярит -> простой русский» доступен;
  2. builtin TM-пары (синтетика, scripts/import/builtin_data.py) — fallback,
     помечается в metadata.source = "synthetic_builtin".

Нормализация: убирает пустые и слишком длинные пары
(EVAL_MAX_SRC_LEN / EVAL_MAX_TGT_LEN из настроек), балансирует по доменам
при сэмплировании (round-robin).

Результат: <out_dir>/bureau-plain_<size>.json для размеров
10/20/50/100/200/500/1000 + manifest.json.
Формат файла:
  {"sources": [...], "references": [...], "domains": [...],
   "source_lang": "bureau", "target_lang": "plain",
   "metadata": {"size": N, "source": "...", "seed": 42}}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SIZES = [10, 20, 50, 100, 200, 500, 1000]
DEFAULT_OUT = REPO_ROOT / "data" / "eval"
DOMAINS = ["general", "legal", "gov", "finance", "hr", "medical"]


def _limits() -> tuple[int, int]:
    from backend.core.config import get_settings

    settings = get_settings()
    return (
        getattr(settings, "EVAL_MAX_SRC_LEN", 2000),
        getattr(settings, "EVAL_MAX_TGT_LEN", 2000),
    )


def _normalize(
    pairs: list[dict[str, str]], max_src: int, max_tgt: int
) -> list[dict[str, str]]:
    seen: set[str] = set()
    result = []
    for p in pairs:
        src, tgt = p["source_text"].strip(), p["target_text"].strip()
        if not src or not tgt or len(src) > max_src or len(tgt) > max_tgt:
            continue
        if src in seen:
            continue
        seen.add(src)
        result.append({"source_text": src, "target_text": tgt, "domain": p.get("domain", "general")})
    return result


def load_hf_pairs(dataset_name: str, split: str, src_field: str, tgt_field: str) -> list[dict[str, str]]:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(dataset_name, split=split, token=token)
    return [
        {"source_text": row[src_field], "target_text": row[tgt_field], "domain": "general"}
        for row in ds
    ]


def load_builtin_pairs() -> list[dict[str, str]]:
    import importlib.util

    builtin_path = REPO_ROOT / "scripts" / "import" / "builtin_data.py"
    spec = importlib.util.spec_from_file_location("builtin_data", builtin_path)
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    return [
        {"source_text": s, "target_text": t, "domain": "general"}
        for s, t in data.generate_tm_pairs()
    ]


def balanced_sample(pairs: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    """Round-robin по доменам: ни один домен не доминирует в выборке."""
    rng = random.Random(seed)
    by_domain: dict[str, list[dict[str, str]]] = {}
    for p in pairs:
        by_domain.setdefault(p["domain"], []).append(p)
    for group in by_domain.values():
        rng.shuffle(group)
    result: list[dict[str, str]] = []
    domains = sorted(by_domain)
    idx = 0
    while len(result) < min(size, len(pairs)):
        domain = domains[idx % len(domains)]
        if by_domain[domain]:
            result.append(by_domain[domain].pop())
        idx += 1
        if all(not g for g in by_domain.values()):
            break
    rng.shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-dataset", default=None, help="имя датасета на HuggingFace")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--src-field", default="source")
    parser.add_argument("--tgt-field", default="target")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hf_dataset:
        source = f"huggingface:{args.hf_dataset}"
        pairs = load_hf_pairs(args.hf_dataset, args.hf_split, args.src_field, args.tgt_field)
    else:
        source = "synthetic_builtin"
        pairs = load_builtin_pairs()

    max_src, max_tgt = _limits()
    pairs = _normalize(pairs, max_src, max_tgt)
    print(f"Нормализовано пар: {len(pairs)} (источник: {source})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"source": source, "seed": args.seed, "files": []}
    for size in SIZES:
        if size > len(pairs):
            print(f"  пропуск size={size}: данных меньше")
            continue
        sample = balanced_sample(pairs, size, args.seed)
        payload = {
            "sources": [p["source_text"] for p in sample],
            "references": [p["target_text"] for p in sample],
            "domains": [p["domain"] for p in sample],
            "source_lang": "bureau",
            "target_lang": "plain",
            "metadata": {"size": len(sample), "source": source, "seed": args.seed},
        }
        path = out_dir / f"bureau-plain_{size}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["files"].append({"size": len(sample), "file": path.name})
        print(f"  записан {path.name}: {len(sample)} пар")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest.json записан в {out_dir}")


if __name__ == "__main__":
    main()
