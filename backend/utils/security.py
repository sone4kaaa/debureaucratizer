from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from backend.core.config import get_settings

_BCRYPT_MAX_BYTES = 72


def _encode_password(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_BYTES:
        raise ValueError(
            f"Пароль длиннее {_BCRYPT_MAX_BYTES} байт в UTF-8 — лимит bcrypt"
        )
    return encoded


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(_encode_password(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            _encode_password(plain_password), hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def create_access_token(data: dict[str, Any]) -> str:
    settings = get_settings()
    for key, value in data.items():
        if not isinstance(value, (str, int, bool)):
            raise ValueError(f"Claim {key!r} должен быть str/int/bool, не {type(value)}")
    now = datetime.now(UTC)
    payload = {
        **data,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Токен истёк") from None
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Неверный токен") from None
