import pytest
from pydantic import ValidationError

from backend.core.config import Settings, get_settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.PROJECT_NAME == "Debureaucratizer"
    assert s.TRANSLATION_ENGINE == "auto"
    assert s.AGENT_ENGINE == "auto"


def test_engine_validator_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, TRANSLATION_ENGINE="google")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, AGENT_ENGINE="magic")


def test_async_database_url_postgres() -> None:
    s = Settings(_env_file=None, DATABASE_URL="postgresql://u:p@db:5432/app")
    assert s.ASYNC_DATABASE_URL == "postgresql+asyncpg://u:p@db:5432/app"


def test_async_database_url_sqlite() -> None:
    s = Settings(_env_file=None, DATABASE_URL="sqlite:///./data/app.db")
    assert s.ASYNC_DATABASE_URL == "sqlite+aiosqlite:///./data/app.db"


def test_async_database_url_passthrough() -> None:
    s = Settings(_env_file=None, DATABASE_URL="sqlite+aiosqlite:///./x.db")
    assert s.ASYNC_DATABASE_URL == "sqlite+aiosqlite:///./x.db"


def test_production_requires_strong_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DEBUG=False, JWT_SECRET_KEY="short")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DEBUG=False)  # дефолтный change-me-in-production

    ok = Settings(_env_file=None, DEBUG=False, JWT_SECRET_KEY="x" * 32)
    assert ok.DEBUG is False


def test_list_properties() -> None:
    s = Settings(
        _env_file=None,
        ENSEMBLE_MODELS="qwen2.5:7b, llama3:8b ,",
        CORS_ORIGINS="http://a, http://b",
        DEBUG=False,
        JWT_SECRET_KEY="x" * 32,
    )
    assert s.ensemble_models_list == ["qwen2.5:7b", "llama3:8b"]
    assert s.cors_origins_list == ["http://a", "http://b"]


def test_cors_debug_allows_all() -> None:
    s = Settings(_env_file=None, DEBUG=True)
    assert s.cors_origins_list == ["*"]


def test_get_settings_cached() -> None:
    assert get_settings() is get_settings()
