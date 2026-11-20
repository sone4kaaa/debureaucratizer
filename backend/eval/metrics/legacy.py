"""Классические метрики: BLEU + chrF++ (sacrebleu)."""


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu

    score = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(score.score, 2)


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu

    score = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return round(score.score, 2)
