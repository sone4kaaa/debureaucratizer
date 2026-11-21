"""Человекочитаемые метки 8 ablation-режимов (для отчётов run_eval)."""

MODE_LABELS: dict[str, str] = {
    "baseline": "LLM only (без глоссария, TM и агентов)",
    "glossary": "LLM + глоссарий",
    "tm": "LLM + глоссарий + TM (BM25)",
    "tm_hybrid": "LLM + глоссарий + TM (BM25 + vector, RRF)",
    "agents_only": "LLM + Critic + Editor (без RAG)",
    "tm_hybrid_critic": "LLM + глоссарий + hybrid TM + Critic",
    "tm_hybrid_editor": "LLM + глоссарий + hybrid TM + Editor",
    "full": "Всё включено (глоссарий + hybrid TM + Critic + Editor)",
}
