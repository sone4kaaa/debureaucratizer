"""Нейросетевые метрики: BERTScore + METEOR."""


def compute_bertscore(
    hypotheses: list[str], references: list[str], lang: str = "ru"
) -> dict[str, float]:
    from bert_score import score as bert_score

    _, _, f1 = bert_score(hypotheses, references, lang=lang, verbose=False)
    return {"bertscore_f1": round(float(f1.mean()), 4)}


def compute_meteor(hypotheses: list[str], references: list[str]) -> float:
    import evaluate

    metric = evaluate.load("meteor")
    result = metric.compute(predictions=hypotheses, references=references)
    return round(float(result["meteor"]), 4)
