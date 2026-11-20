"""Запасной движок: HuggingFace Transformers (Seq2Seq и Causal LM)."""

import asyncio

import torch

from backend.core.config import get_settings
from backend.core.constants import DEFAULT_SEED
from backend.core.exceptions import ModelNotLoadedError
from backend.utils.logging import logger


class TranslatorHuggingFace:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().HF_MODEL
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoConfig, AutoTokenizer

        logger.info("Загрузка HF-модели %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if getattr(config, "is_encoder_decoder", False):
            from transformers import AutoModelForSeq2SeqLM

            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, dtype=dtype)
        else:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()

    async def translate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        return await asyncio.to_thread(self._translate_sync, prompt, max_new_tokens)

    def _translate_sync(self, prompt: str, max_new_tokens: int) -> str:
        if self._model is None:
            try:
                self._load()
            except Exception as exc:
                raise ModelNotLoadedError(
                    f"Не удалось загрузить HF-модель {self.model_name}: {exc}"
                ) from exc
        torch.manual_seed(DEFAULT_SEED)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
