"""HTTP-клиент Streamlit UI к FastAPI бэкенду + хелперы авторизации."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_URL: str = os.environ.get("API_URL", "http://localhost:8000")

EVAL_MODES: list[str] = [
    "full",
    "baseline",
    "glossary",
    "tm",
    "tm_hybrid",
    "agents_only",
    "tm_hybrid_critic",
    "tm_hybrid_editor",
]
DOMAINS: list[str] = ["general", "legal", "gov", "finance", "hr", "medical"]
ENGINES: list[str] = ["auto", "ollama", "huggingface"]


class APIError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


def _headers() -> dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_request(
    method: str,
    path: str,
    json_body: dict | list | None = None,
    params: dict | None = None,
    files: dict | None = None,
    timeout: float = 330.0,
) -> dict:
    """Вызов API. Бросает APIError с detail из ответа при ошибке."""
    resp = requests.request(
        method,
        f"{API_URL}{path}",
        json=json_body,
        params=params,
        files=files,
        headers=_headers(),
        timeout=timeout,
    )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise APIError(resp.status_code, str(detail))
    return resp.json()


def get_styles() -> dict[str, str]:
    """Справочник стилей с /languages; кэшируется в session_state."""
    if "styles" not in st.session_state:
        try:
            st.session_state.styles = api_request("GET", "/languages")["styles"]
        except Exception:
            st.session_state.styles = {
                "bureau": "Канцелярит (официально-деловой)",
                "legal": "Юридический язык",
                "official": "Официальные уведомления госорганов",
                "plain": "Простой русский",
            }
    return st.session_state.styles


def style_selectbox(label: str, key: str, default: str = "bureau") -> str:
    styles = get_styles()
    options = list(styles.keys())
    labels = [f"{code} — {name}" for code, name in styles.items()]
    idx = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=idx, format_func=lambda c: f"{c} — {styles[c]}", key=key)


def is_logged_in() -> bool:
    return bool(st.session_state.get("token"))


def current_role() -> str | None:
    return (st.session_state.get("user") or {}).get("role")


def require_auth() -> str | None:
    """Вернуть токен или показать предупреждение и остановить страницу."""
    if not is_logged_in():
        st.warning("Сначала войдите на главной странице.")
        st.stop()
    return st.session_state["token"]


def require_role(roles: tuple[str, ...]) -> None:
    require_auth()
    if current_role() not in roles:
        st.error(f"Нужна роль: {' / '.join(roles)}. Ваша роль: {current_role() or 'неизвестна'}.")
        st.stop()
