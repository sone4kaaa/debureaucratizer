"""Страница «Оценка качества»: BLEU / chrF++ / BERTScore."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import api_request, require_auth  # noqa: E402

st.set_page_config(page_title="Оценка качества", layout="wide")
require_auth()

st.header("Оценка качества переводов")
st.caption("По одному варианту в строке: hypotheses — ваши переводы, references — эталоны.")

c1, c2 = st.columns(2)
hypotheses_raw = c1.text_area(
    "Hypotheses (ваши упрощения)",
    height=200,
    placeholder="Сообщаем: нужно оплатить до 15 марта 2025 года.\nВ услуге откажут, если нет паспорта.",
)
references_raw = c2.text_area(
    "References (эталонные упрощения)",
    height=200,
    placeholder="Нужно оплатить до 15 марта 2025 года.\nБез паспорта в услуге откажут.",
)
include_neural = st.checkbox("Нейросетевые метрики (BERTScore, требует bert-score)", value=False)

if st.button("Оценить", type="primary", use_container_width=True):
    hypotheses = [line.strip() for line in hypotheses_raw.splitlines() if line.strip()]
    references = [line.strip() for line in references_raw.splitlines() if line.strip()]
    if not hypotheses or len(hypotheses) != len(references):
        st.warning("Списки должны быть непустыми и одинаковой длины (по строкам)")
    else:
        with st.spinner("Считаем метрики…"):
            try:
                metrics = api_request(
                    "POST",
                    "/eval/run",
                    json_body={"hypotheses": hypotheses, "references": references, "include_neural": include_neural},
                )
            except Exception as exc:
                st.error(f"Ошибка: {exc}")
                st.stop()
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            if isinstance(value, (int, float)):
                col.metric(name, value)
            else:
                col.metric(name, "—")
        with st.expander("Полный JSON"):
            st.json(metrics)
