"""Страница «Глоссарий»: справочник канцеляризмов (роль admin)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import api_request, require_role, style_selectbox  # noqa: E402

st.set_page_config(page_title="Глоссарий", layout="wide")
require_role(("admin",))

st.header("Глоссарий: канцеляризм -> простой эквивалент")

col_view, col_edit = st.columns([2, 1])

with col_view:
    st.subheader("Просмотр")
    source_lang = style_selectbox("Фильтр: исходный", "gl_src", default="bureau")
    page_size = st.select_slider("Строк на странице", [20, 50, 100, 200], value=50)
    page = st.number_input("Страница", min_value=1, value=1, step=1)
    try:
        data = api_request(
            "GET",
            "/admin/glossary/list",
            params={
                "source_lang": source_lang,
                "target_lang": "plain",
                "limit": page_size,
                "offset": (page - 1) * page_size,
            },
        )
    except Exception as exc:
        st.error(f"Ошибка: {exc}")
        st.stop()

    st.caption(f"Всего записей: {data.get('total')}")
    if data.get("items"):
        st.dataframe(
            [{"канцеляризм": i["source_text"], "простой вариант": i["target_text"]} for i in data["items"]],
            use_container_width=True,
        )
    if st.button("Обновить"):
        st.rerun()

with col_edit:
    st.subheader("Добавить запись")
    with st.form("add_glossary"):
        src = st.text_input("Канцеляризм", key="gl_add_src")
        tgt = st.text_input("Простой эквивалент", key="gl_add_tgt")
        if st.form_submit_button("Добавить"):
            if src.strip() and tgt.strip():
                try:
                    api_request(
                        "POST",
                        "/admin/glossary/add",
                        json_body={"source_lang": "bureau", "target_lang": "plain", "source_text": src, "target_text": tgt},
                    )
                    st.success("Добавлено")
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")
            else:
                st.warning("Заполните оба поля")

    st.subheader("Удалить по ID")
    del_id = st.text_input("ID записи", key="gl_del_id")
    if st.button("Удалить"):
        try:
            api_request("DELETE", f"/admin/glossary/delete/{del_id}")
            st.success("Удалено")
        except Exception as exc:
            st.error(f"Ошибка: {exc}")

    st.subheader("Импорт CSV")
    st.caption("Формат: source_lang,target_lang,канцеляризм,простой вариант (первая строка — заголовок)")
    uploaded = st.file_uploader("CSV-файл", type=["csv"])
    if uploaded is not None and st.button("Загрузить CSV"):
        try:
            result = api_request(
                "POST",
                "/admin/glossary/import-csv",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            )
            st.success(f"Добавлено: {result.get('added')} из {result.get('total')}")
        except Exception as exc:
            st.error(f"Ошибка: {exc}")
