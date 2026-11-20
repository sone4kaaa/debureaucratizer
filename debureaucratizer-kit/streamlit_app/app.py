"""Главная страница: вход / регистрация + статус сервиса."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.api import API_URL, api_request, current_role, is_logged_in  # noqa: E402

st.set_page_config(page_title="Дебюрократизатор", page_icon="", layout="wide")

# --- Боковая панель ---
with st.sidebar:
    st.title("Дебюрократизатор")
    st.caption("Канцелярит -> простой русский")
    if is_logged_in():
        user = st.session_state.get("user") or {}
        st.markdown(
            f"Вы вошли как **{user.get('username', '?')}** (роль: `{current_role()}`)"
        )
        if st.button("Выйти"):
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()
    else:
        st.info("Войдите, чтобы переводить тексты")
    st.divider()
    st.caption(f"API: {API_URL}")
    health = None
    try:
        health = api_request("GET", "/healthz", timeout=5.0)
    except Exception:
        pass
    if health:
        st.success("API доступен")
    else:
        st.error("API недоступен — запустите uvicorn")

# --- Главная область ---
st.header("Дебюрократизатор")
st.markdown(
    "Система переписывает тексты с канцелярского и юридического русского "
    "на простой язык с сохранением смысла, фактов, дат и сумм: "
    "письма госорганов, договоры, уведомления, предписания."
)

if not is_logged_in():
    tab_login, tab_register = st.tabs(["Вход", "Регистрация"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Пароль", type="password")
            submit = st.form_submit_button("Войти", use_container_width=True)
        if submit:
            try:
                data = api_request(
                    "POST", "/auth/login", json_body={"email": email, "password": password}
                )
                st.session_state.token = data["access_token"]
                st.session_state.user = api_request("GET", "/auth/me")
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка входа: {exc}")

    with tab_register:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            username = st.text_input("Имя пользователя", key="reg_username")
            reg_password = st.text_input("Пароль (мин. 8 символов)", type="password", key="reg_pass")
            submit = st.form_submit_button("Создать аккаунт", use_container_width=True)
        if submit:
            try:
                data = api_request(
                    "POST",
                    "/auth/register",
                    json_body={
                        "email": reg_email,
                        "username": username,
                        "password": reg_password,
                    },
                )
                st.session_state.token = data["access_token"]
                st.session_state.user = api_request("GET", "/auth/me")
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка регистрации: {exc}")
else:
    st.success("Вы авторизованы. Выберите страницу в меню слева.")
    st.markdown(
        """
        - **Перевод** — упрощение одного текста (глоссарий + RAG + агенты)
        - **Мульти и батч** — один текст в несколько стилей / пачка текстов
        - **Ensemble** — несколько моделей, критик выбирает лучший вариант
        - **Глоссарий** и **Память переводов** — справочники (нужна роль admin)
        - **Оценка качества** — BLEU / chrF / BERTScore по вашим парам
        - **HITL** — разбор спорных переводов (нужна роль editor/admin)
        - **Админ** — статистика и очистка кэша
        """
    )
