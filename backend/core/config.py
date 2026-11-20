from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALLOWED_ENGINES = {"auto", "ollama", "huggingface"}
_DEFAULT_JWT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    # Приложение
    PROJECT_NAME: str = "Debureaucratizer"
    DEBUG: bool = True
    API_PORT: int = 8000

    # База данных
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # Auth
    JWT_SECRET_KEY: str = _DEFAULT_JWT_SECRET
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 1440

    # LLM: Ollama
    OLLAMA_HOST: str = "localhost"
    OLLAMA_PORT: int = 11434
    OLLAMA_MODEL: str = "qwen2.5:7b"
    TRANSLATION_ENGINE: str = "auto"
    AGENT_ENGINE: str = "auto"
    FALLBACK_TO_HF_ON_ERROR: bool = True

    # LLM: HuggingFace
    HF_MODEL: str = "facebook/nllb-200-distilled-600M"
    HF_AGENT_MODEL: str = ""
    HF_TOKEN: str = ""

    # Ensemble
    ENSEMBLE_MODELS: str = "qwen2.5:7b"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "tm_entries"

    # Redis / Celery
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Режимы
    EVAL_MODE: str = "full"
    CACHEABLE_DOMAIN: str = "general"

    # CORS
    CORS_ORIGINS: str = "http://localhost:8501"

    # Логирование
    LOG_JSON: bool = False

    @field_validator("TRANSLATION_ENGINE", "AGENT_ENGINE")
    @classmethod
    def _validate_engine(cls, v: str) -> str:
        if v not in _ALLOWED_ENGINES:
            raise ValueError(
                f"Недопустимый движок {v!r}, допустимы: {sorted(_ALLOWED_ENGINES)}"
            )
        return v

    @model_validator(mode="after")
    def _validate_production(self) -> "Settings":
        if not self.DEBUG:
            if (
                self.JWT_SECRET_KEY == _DEFAULT_JWT_SECRET
                or len(self.JWT_SECRET_KEY) < 32
            ):
                raise ValueError(
                    "При DEBUG=false задайте JWT_SECRET_KEY длиной >= 32 символов "
                    "(например: openssl rand -hex 32)"
                )
        return self

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("sqlite:///") and not url.startswith("sqlite+aiosqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        return url

    @property
    def ollama_url(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"

    @property
    def ensemble_models_list(self) -> list[str]:
        return [m.strip() for m in self.ENSEMBLE_MODELS.split(",") if m.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        if self.DEBUG:
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
