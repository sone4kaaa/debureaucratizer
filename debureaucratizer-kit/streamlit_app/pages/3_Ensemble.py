"""Страница «Ensemble»: несколько моделей, критик выбирает лучшую."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import DOMAINS, api_request, require_auth, style_selectbox  # noqa: E402

st.set_page_config(page_title="Ensemble", layout="wide")
require_auth()

st.header("Ensemble: ансамбль моделей + критик")
st.caption("Модели из Ollama. Чем выше score критика, тем лучше вариант.")

source_lang = style_selectbox("Исходный стиль", "en_src", default="bureau")
domain = st.selectbox("Домен", DOMAINS, index=0, key="en_dom")
models_raw = st.text_input("Модели через запятую (пусто = из .env)", placeholder="qwen2.5:7b, qwen2.5:3b")
text = st.text_area("Текст", height=180, key="en_text")

if st.button("Собрать ансамбль", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Введите текст")
    else:
        models = [m.strip() for m in models_raw.split(",") if m.strip()] or None
        with st.spinner("Генерируем варианты и оцениваем критиком…"):
            try:
                result = api_request(
                    "POST",
                    "/translate/ensemble",
                    json_body={
                        "text": text,
                        "source_lang": source_lang,
                        "target_lang": "plain",
                        "domain": domain,
                        "models": models,
                    },
                )
            except Exception as exc:
                st.error(f"Ошибка: {exc}")
                st.stop()

        best = result.get("best")
        if best:
            st.subheader("Лучший вариант (по критику)")
            st.success(best.get("text", ""))
            st.caption(f"Модель: {best.get('model')} | score: {best.get('critic_score')}")

        st.subheader("Все варианты")
        for v in result.get("variants") or []:
            with st.expander(f"{v.get('model')} — score: {v.get('critic_score')}"):
                st.write(v.get("text", ""))
