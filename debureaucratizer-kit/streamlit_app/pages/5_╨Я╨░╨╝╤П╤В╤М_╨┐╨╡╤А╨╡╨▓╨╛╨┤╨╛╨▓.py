"""Страница «Память переводов»: пары «сложное -> простое» (роль admin)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import api_request, require_role, style_selectbox  # noqa: E402

st.set_page_config(page_title="Память переводов", layout="wide")
require_role(("admin",))

st.header("Translation Memory: сложное предложение -> простое")

st.subheader("Просмотр")
source_lang = style_selectbox("Фильтр: исходный", "tm_src", default="bureau")
page_size = st.select_slider("Строк на странице", [10, 25, 50, 100], value=25, key="tm_ps")
page = st.number_input("Страница", min_value=1, value=1, step=1, key="tm_page")
try:
    data = api_request(
        "GET",
        "/admin/tm/list",
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
        [
            {
                "сложное": i["source_text"],
                "простое": i["target_text"],
                "id": i["id"],
            }
            for i in data["items"]
        ],
        use_container_width=True,
    )

c1, c2 = st.columns(2)
with c1:
    st.subheader("Добавить пару")
    with st.form("add_tm"):
        src = st.text_area("Сложное предложение", height=80, key="tm_add_src")
        tgt = st.text_area("Простое предложение", height=80, key="tm_add_tgt")
        if st.form_submit_button("Добавить"):
            if src.strip() and tgt.strip():
                try:
                    api_request(
                        "POST",
                        "/admin/tm/add",
                        json_body={"source_lang": "bureau", "target_lang": "plain", "source_text": src, "target_text": tgt},
                    )
                    st.success("Добавлено. Переиндексация — через Celery или перезапуск API.")
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")
            else:
                st.warning("Заполните оба поля")

with c2:
    st.subheader("Импорт CSV")
    st.caption("Формат: source_lang,target_lang,сложное,простое")
    uploaded = st.file_uploader("CSV-файл", type=["csv"], key="tm_csv")
    if uploaded is not None and st.button("Загрузить CSV"):
        try:
            result = api_request(
                "POST",
                "/admin/tm/import-csv",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            )
            st.success(f"Добавлено: {result.get('added')} из {result.get('total')}")
        except Exception as exc:
            st.error(f"Ошибка: {exc}")
