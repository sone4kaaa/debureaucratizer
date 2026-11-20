"""SentenceTransformer-эмбеддинги: синглтон с ленивой загрузкой."""

import asyncio

import numpy as np

from backend.utils.logging import logger

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Загрузка embedding-модели %s", MODEL_NAME)
            self._model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Батч-кодирование: один вызов encode() на весь список."""
        self._load()
        return self._model.encode(texts, convert_to_numpy=True)

    async def encode_async(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.encode, texts)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
