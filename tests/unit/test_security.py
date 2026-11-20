import time

import jwt
import pytest
from fastapi import HTTPException

from backend.core.config import get_settings
from backend.utils.security import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)


def test_hash_and_verify() -> None:
    hashed = get_password_hash("Secret123")
    assert hashed != "Secret123"
    assert verify_password("Secret123", hashed)
    assert not verify_password("Wrong123", hashed)


def test_password_over_72_bytes_rejected() -> None:
    long_pw = "а" * 40  # 80 байт в UTF-8
    with pytest.raises(ValueError):
        get_password_hash(long_pw)
    assert not verify_password(long_pw, get_password_hash("Secret123"))


def test_token_roundtrip() -> None:
    token = create_access_token({"sub": "user-1", "role": "admin"})
    payload = decode_token(token)
    assert payload["sub"] == "user-1"
    assert payload["role"] == "admin"
    assert "exp" in payload and "iat" in payload


def test_expired_token() -> None:
    settings = get_settings()
    token = jwt.encode(
        {"sub": "u", "exp": int(time.time()) - 10},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_invalid_token() -> None:
    with pytest.raises(HTTPException) as exc:
        decode_token("not-a-token")
    assert exc.value.status_code == 401


def test_claim_type_validation() -> None:
    with pytest.raises(ValueError):
        create_access_token({"sub": "u", "nested": {"a": 1}})
