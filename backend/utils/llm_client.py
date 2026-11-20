"""Вызов Ollama HTTP API."""

from backend.core.config import get_settings
from backend.core.constants import DEFAULT_SEED
from backend.core.exceptions import EngineUnavailableError
from backend.utils.http_client import get_http_client


async def ollama_generate(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    seed: int = DEFAULT_SEED,
    max_tokens: int = 1024,
) -> str:
    settings = get_settings()
    client = get_http_client(timeout=300.0)
    try:
        resp = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": model or settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_predict": max_tokens,
                },
            },
        )
        resp.raise_for_status()
    except Exception as exc:
        raise EngineUnavailableError(f"Ollama недоступна: {exc}") from exc
    return resp.json().get("response", "")
