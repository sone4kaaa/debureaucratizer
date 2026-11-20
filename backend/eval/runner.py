"""Evaluator — фасад над метриками."""

from typing import Any

from backend.eval.metrics.legacy import compute_bleu, compute_chrf
from backend.eval.metrics.neural import compute_bertscore, compute_meteor
from backend.utils.logging import logger


class Evaluator:
    def run(
        self,
        hypotheses: list[str],
        references: list[str],
        lang: str = "ru",
        include_neural: bool = True,
    ) -> dict[str, Any]:
        if not hypotheses or len(hypotheses) != len(references):
            raise ValueError("hypotheses и references должны быть непустыми и равной длины")
        result: dict[str, Any] = {
            "count": len(hypotheses),
            "bleu": compute_bleu(hypotheses, references),
            "chrf++": compute_chrf(hypotheses, references),
        }
        if include_neural:
            try:
                result.update(compute_bertscore(hypotheses, references, lang=lang))
            except Exception as exc:
                logger.warning("BERTScore не посчитан: %s", exc)
            try:
                result["meteor"] = compute_meteor(hypotheses, references)
            except Exception as exc:
                logger.warning("METEOR не посчитан: %s", exc)
        return result


evaluator = Evaluator()
