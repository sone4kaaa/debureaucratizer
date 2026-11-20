"""Страница «Перевод»: упрощение одного текста."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import DOMAINS, ENGINES, EVAL_MODES, api_request, require_auth, style_selectbox  # noqa: E402

st.set_page_config(page_title="Перевод", layout="wide")
require_auth()

st.header("Перевод: канцелярит -> простой русский")

col_src, col_tgt = st.columns(2)
source_lang = style_selectbox("Исходный стиль", "tr_src", default="bureau")
target_lang = style_selectbox("Целевой стиль", "tr_tgt", default="plain")

col_mode, col_dom, col_eng = st.columns(3)
eval_mode = col_mode.selectbox("Режим пайплайна", EVAL_MODES, index=0)
domain = col_dom.selectbox("Домен", DOMAINS, index=0)
engine = col_eng.selectbox("Движок", ENGINES, index=0)

text = st.text_area(
    "Текст на «официальном» языке",
    placeholder="В соответствии с вышеизложенным уведомляем о необходимости осуществления оплаты в срок до 15 марта 2025 года…",
    height=200,
)

if st.button("Упростить", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Введите текст")
    else:
        with st.spinner("Переводим (агенты + RAG, может занять до минуты)…"):
            try:
                result = api_request(
                    "POST",
                    "/translate",
                    json_body={
                        "text": text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "domain": domain,
                        "eval_mode": eval_mode,
                        "engine": None if engine == "auto" else engine,
                    },
                )
            except Exception as exc:
                st.error(f"Ошибка: {exc}")
                st.stop()

        st.subheader("Результат")
        st.success(result.get("translated_text", ""))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Оценка критика", result.get("critic_score"), delta=None if result.get("critic_passed") is None else ("ок" if result.get("critic_passed") else "не прошёл"))
        m2.metric("Валидатор", "пройден" if result.get("validation_passed") else "с замечаниями")
        m3.metric("Движок", f"{result.get('engine_used')}")
        m4.metric("Из кэша", "да" if result.get("from_cache") else "нет")

        if result.get("validator_errors") or result.get("validator_warnings"):
            with st.expander("Замечания валидатора"):
                for e in result.get("validator_errors") or []:
                    st.error(e)
                for w in result.get("validator_warnings") or []:
                    st.warning(w)
        if result.get("critic_suggestions"):
            with st.expander("Советы критика"):
                for s in result.get("critic_suggestions"):
                    st.write(f"- {s}")
        if result.get("editor_changed"):
            with st.expander("Редактор"):
                st.write(f"Правки применены. Причина отката: {result.get('editor_rolled_back')}")
        if result.get("reasoning_log"):
            with st.expander("Журнал агентов"):
                for line in result.get("reasoning_log"):
                    st.caption(str(line))
        with st.expander("Использованные примеры RAG"):
            st.json(
                {
                    "glossary": result.get("glossary_used") or [],
                    "tm_examples": result.get("tm_examples") or [],
                }
            )
        with st.expander("Полный JSON ответа"):
            st.json(result)
