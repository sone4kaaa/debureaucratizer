"""Страница «Админ»: кэш переводов и служебная информация (роль admin)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.api import API_URL, api_request, require_role  # noqa: E402

st.set_page_config(page_title="Админ", layout="wide")
require_role(("admin",))

st.header("Администрирование")

col_api, col_ui = st.columns(2)
col_api.metric("Адрес API", API_URL)

health = None
try:
    health = api_request("GET", "/healthz", timeout=5.0)
except Exception:
    pass
col_api.metric("API", "доступен" if health else "недоступен")

st.subheader("Кэш переводов")
try:
    stats = api_request("GET", "/admin/cache/stats")
except Exception as exc:
    stats = {}
    st.error(f"Ошибка: {exc}")
if stats:
    st.json(stats)

clear_mode = st.text_input("Очистить только режим (пусто = весь кэш)", placeholder="full")
if st.button("Очистить кэш", type="primary"):
    try:
        params = {"eval_mode": clear_mode.strip()} if clear_mode.strip() else None
        result = api_request("DELETE", "/admin/cache/clear", params=params)
        st.success(f"Удалено записей: {result.get('deleted')}")
    except Exception as exc:
        st.error(f"Ошибка: {exc}")

st.divider()
st.subheader("Подсказка")
st.markdown(
    """
    Роли пользователей меняются вручную в БД или скриптом:

    ```bash
    python -m poetry run python scripts/import/import_builtin.py --admin <username>
    ```
    """
)
