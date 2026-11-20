"""Страница «HITL»: разбор спорных переводов (роль editor/admin)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import api_request, require_role  # noqa: E402

st.set_page_config(page_title="HITL", layout="wide")
require_role(("admin", "editor"))

st.header("Human-in-the-loop: спорные переводы")

try:
    stats = api_request("GET", "/hitl/uncertain/stats")
except Exception as exc:
    st.error(f"Ошибка: {exc}")
    st.stop()

m = st.columns(len(stats))
for col, (name, value) in zip(m, stats.items()):
    col.metric(name, value)

st.divider()

if st.button("Обновить список", use_container_width=True):
    st.rerun()

try:
    data = api_request("GET", "/hitl/uncertain", params={"limit": 20})
except Exception as exc:
    st.error(f"Ошибка: {exc}")
    st.stop()

items = data.get("items") or []
if not items:
    st.success("Очередь пуста: спорных переводов нет.")
    st.stop()

for row in items:
    with st.expander(
        f"score: {row.get('critic_score')} | валидатор: "
        f"{'ок' if row.get('validation_passed') else 'ошибки'} | {row.get('created_at') or ''}"
    ):
        st.markdown("**Источник:**")
        st.write(row.get("source_text", ""))
        st.markdown("**Машинный перевод:**")
        st.info(row.get("machine_translation", ""))
        st.markdown(f"**ID:** `{row.get('id')}`")
        corrected = st.text_input("Исправленный вариант (пусто = подтвердить как есть)", key=f"corr_{row['id']}")
        c1, c2 = st.columns(2)
        if c1.button("Подтвердить", key=f"ok_{row['id']}"):
            try:
                api_request(
                    "POST",
                    f"/hitl/uncertain/{row['id']}/confirm",
                    json_body={"corrected_translation": corrected.strip() or None},
                )
                st.success("Подтверждено и добавлено в TM")
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка: {exc}")
        if c2.button("Отклонить", key=f"rej_{row['id']}"):
            try:
                api_request("POST", f"/hitl/uncertain/{row['id']}/reject")
                st.success("Отклонено")
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка: {exc}")
