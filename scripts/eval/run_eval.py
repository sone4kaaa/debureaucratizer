"""Прогон оценки по 8 ablation-режимам.

Читает датасет (см. prepare_eval_datasets.py), для каждого режима вызывает
perform_translation(strict_engine=True) и считает метрики Evaluator'ом.

strict_engine=True: если Ollama недоступна и запрошен ollama, прогон падает
с EngineFallbackTriggeredError — retrieval-режимы не должны незаметно
деградировать в baseline на HF-движке.

Использование (из корня проекта, API не нужен, нужны БД и Ollama):
  python scripts/eval/run_eval.py data/eval/bureau-plain_10.json
  python scripts/eval/run_eval.py data/eval/bureau-plain_10.json --modes baseline full
  python scripts/eval/run_eval.py ... --no-neural --out reports/eval_run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.mode_labels import MODE_LABELS  # noqa: E402


async def run_mode(
    session,
    sources: list[str],
    mode: str,
    engine: str | None,
) -> list[str]:
    from backend.services.pipeline import perform_translation

    hypotheses: list[str] = []
    for i, source in enumerate(sources, 1):
        result = await perform_translation(
            session,
            source,
            eval_mode=mode,
            engine=engine,
            strict_engine=True,
            use_cache=False,  # честный прогон, без кэша прошлых запусков
        )
        hypotheses.append(result["translated_text"])
        if i % 10 == 0 or i == len(sources):
            print(f"    {mode}: {i}/{len(sources)}", flush=True)
    return hypotheses


async def main_async(args) -> dict:
    from backend.db.models import Base
    from backend.db.session import AsyncSessionLocal, engine
    from backend.eval.runner import evaluator

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    sources: list[str] = dataset["sources"]
    references: list[str] = dataset["references"]
    modes = args.modes or list(MODE_LABELS)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    report: dict = {
        "dataset": str(args.dataset),
        "size": len(sources),
        "engine": args.engine or "auto",
        "modes": {},
    }
    async with AsyncSessionLocal() as session:
        for mode in modes:
            print(f"Режим {mode} ({MODE_LABELS.get(mode, mode)})")
            started = time.monotonic()
            hypotheses = await run_mode(session, sources, mode, args.engine)
            metrics = evaluator.run(
                hypotheses, references, include_neural=not args.no_neural
            )
            metrics["label"] = MODE_LABELS.get(mode, mode)
            metrics["seconds"] = round(time.monotonic() - started, 1)
            report["modes"][mode] = metrics
            print(f"    -> {metrics}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="путь к JSON от prepare_eval_datasets.py")
    parser.add_argument("--modes", nargs="+", default=None, help="подмножество режимов")
    parser.add_argument("--engine", default=None, help="ollama | huggingface")
    parser.add_argument("--no-neural", action="store_true", help="только BLEU/chrF++")
    parser.add_argument("--out", default=None, help="куда записать JSON-отчёт")
    args = parser.parse_args()

    report = asyncio.run(main_async(args))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Отчёт записан в {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
