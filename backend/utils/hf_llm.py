"""HuggingFace-обёртка LLM для агентов: единый интерфейс generate()."""

import asyncio

import torch

from backend.core.constants import DEFAULT_SEED
from backend.utils.logging import logger


class HuggingFaceLLM:
    def __init__(self, model_name: str) -> None:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        self.device = device

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "is_encoder_decoder", False):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        logger.info("HF-модель %s загружена на %s (%s)", model_name, device, dtype)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = DEFAULT_SEED,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, prompt, max_new_tokens, temperature, seed
        )

    def _generate_sync(
        self, prompt: str, max_new_tokens: int, temperature: float, seed: int
    ) -> str:
        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


_REGISTRY: dict[str, HuggingFaceLLM] = {}


def get_hf_llm(model_name: str) -> HuggingFaceLLM:
    if model_name not in _REGISTRY:
        _REGISTRY[model_name] = HuggingFaceLLM(model_name)
    return _REGISTRY[model_name]


def unload_hf_llm() -> None:
    _REGISTRY.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
