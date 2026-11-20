from abc import ABC, abstractmethod
from typing import Any

from backend.core.config import get_settings
from backend.core.constants import DEFAULT_SEED
from backend.utils.async_utils import retry
from backend.utils.logging import logger


class BaseAgent(ABC):
    """Абстрактный агент: name, последний результат, process(state)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._last_result: dict[str, Any] | None = None

    @abstractmethod
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        ...

    def get_last_result(self) -> dict[str, Any] | None:
        return self._last_result


class Agent(BaseAgent):
    """LLM-backed агент: резолвит движок, вызывает LLM с retry и фиксированным seed."""

    hf_model_setting: str = "HF_AGENT_MODEL"
    ollama_model_setting: str = "OLLAMA_MODEL"
    temperature: float = 0.0
    max_new_tokens: int = 1024
    seed: int = DEFAULT_SEED

    def __init__(self, name: str, engine: str | None = None) -> None:
        super().__init__(name)
        settings = get_settings()
        requested = engine or settings.AGENT_ENGINE
        hf_model = getattr(settings, self.hf_model_setting, "") or settings.HF_MODEL

        if requested == "auto":
            self.engine = "huggingface" if settings.HF_AGENT_MODEL else "ollama"
        elif requested == "huggingface":
            if not hf_model:
                logger.warning("HF-модель не задана, fallback на ollama")
                self.engine = "ollama"
            else:
                self.engine = "huggingface"
        else:
            self.engine = "ollama"

        self.model = (
            hf_model
            if self.engine == "huggingface"
            else getattr(settings, self.ollama_model_setting)
        )

    @retry(max_attempts=3, delay=1.0)
    async def _call_llm(self, prompt: str) -> str:
        """Единая точка вызова LLM. Seed пиннится на каждый вызов."""
        if self.engine == "ollama":
            from backend.utils.llm_client import ollama_generate

            return await ollama_generate(
                prompt,
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_new_tokens,
            )
        from backend.utils.hf_llm import get_hf_llm

        llm = get_hf_llm(self.model)
        return await llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
