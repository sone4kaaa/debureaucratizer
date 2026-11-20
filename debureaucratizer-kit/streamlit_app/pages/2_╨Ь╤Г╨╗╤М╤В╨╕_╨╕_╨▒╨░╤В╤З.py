"""Страница «Мульти и батч»: один текст в несколько стилей или пачка текстов."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import DOMAINS, api_request, require_auth, style_selectbox  # noqa: E402

st.set_page_config(page_title="Мульти и батч", layout="wide")
require_auth()

st.header("Мульти- и батч-перевод")

tab_multi, tab_batch = st.tabs(["Один текст -> несколько стилей", "Пачка текстов"])

with tab_multi:
    source_lang = style_selectbox("Исходный стиль", "mu_src", default="bureau")
    domain = st.selectbox("Домен", DOMAINS, index=0, key="mu_dom")
    target_langs = st.multiselect(
        "Целевые стили",
        ["plain"],
        default=["plain"],
        format_func=lambda c: c,
    )
    text = st.text_area("Текст", height=160, key="mu_text")
    if st.button("Перевести в несколько стилей", use_container_width=True):
        if not text.strip() or not target_langs:
            st.warning("Нужны текст и хотя бы один целевой стиль")
        else:
            with st.spinner("Переводим…"):
                try:
                    result = api_request(
                        "POST",
                        "/translate/multi",
                        json_body={
                            "text": text,
                            "source_lang": source_lang,
                            "target_langs": target_langs,
                            "domain": domain,
                        },
                    )
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")
                    st.stop()
            st.markdown("**Источник:**")
            st.info(text)
            for tgt, res in (result.get("results") or {}).items():
                st.markdown(f"**{tgt}:**")
                st.success(res.get("translated_text", res.get("error", "—")))

with tab_batch:
    source_lang_b = style_selectbox("Исходный стиль", "ba_src", default="bureau")
    domain_b = st.selectbox("Домен", DOMAINS, index=0, key="ba_dom")
    raw = st.text_area(
        "Тексты — по одному на строку", height=200, key="ba_texts",
        placeholder="Уведомляем о необходимости предоставления документов.\nПросим произвести оплату в установленный срок.",
    )
    if st.button("Перевести пачку", use_container_width=True):
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
        if not texts:
            st.warning("Введите хотя бы один текст")
        else:
            with st.spinner(f"Переводим {len(texts)} шт. параллельно…"):
                try:
                    result = api_request(
                        "POST",
                        "/translate/batch",
                        json_body={
                            "texts": texts,
                            "source_lang": source_lang_b,
                            "target_lang": "plain",
                            "domain": domain_b,
                        },
                    )
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")
                    st.stop()
            st.caption(f"Обработано: {result.get('count')}")
            for i, res in enumerate(result.get("results") or [], 1):
                with st.expander(f"{i}. {res.get('source_text', '')[:80]}"):
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        st.success(res.get("translated_text", ""))
