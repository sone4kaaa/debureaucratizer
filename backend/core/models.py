from pydantic import BaseModel, EmailStr, field_validator

from backend.utils.security import _BCRYPT_MAX_BYTES


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Пароль должен быть не короче 8 символов")
        if not any(ch.isdigit() for ch in v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        if not any(ch.isupper() for ch in v):
            raise ValueError("Пароль должен содержать хотя бы одну заглавную букву")
        if len(v.encode("utf-8")) > _BCRYPT_MAX_BYTES:
            raise ValueError(f"Пароль длиннее {_BCRYPT_MAX_BYTES} байт в UTF-8")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: str
