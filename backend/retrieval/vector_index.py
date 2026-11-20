"""Векторный индекс на Qdrant (cosine, фильтры по стилевой паре)."""

import asyncio
import uuid
from typing import Any

from backend.core.config import get_settings
from backend.core.exceptions import RetrievalError
from backend.utils.logging import logger

_POINT_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")
VECTOR_SIZE = 384  # paraphrase-multilingual-MiniLM-L12-v2


def point_id(source_lang: str, target_lang: str, en: str, tgt: str) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{source_lang}|{target_lang}|{en}|{tgt}"))


class VectorIndex:
    def __init__(self) -> None:
        settings = get_settings()
        from qdrant_client import QdrantClient

        self._client = QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
        self._collection = settings.QDRANT_COLLECTION

    def ensure_collection(self) -> None:
        """Создаёт коллекцию при отсутствии. Без drop — безопасно для нескольких процессов."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Создана коллекция Qdrant %s", self._collection)

    def recreate_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    async def add_entries(
        self,
        entries: list[dict[str, Any]],
        vectors: Any,
        source_lang: str,
        target_lang: str,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=point_id(source_lang, target_lang, e["source_text"], e["target_text"]),
                vector=vec.tolist(),
                payload={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_text": e["source_text"],
                    "target_text": e["target_text"],
                },
            )
            for e, vec in zip(entries, vectors)
        ]
        await asyncio.to_thread(
            self._client.upsert, collection_name=self._collection, points=points
        )

    async def search(
        self,
        query_vector: Any,
        source_lang: str,
        target_lang: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            must=[
                FieldCondition(key="source_lang", match=MatchValue(value=source_lang)),
                FieldCondition(key="target_lang", match=MatchValue(value=target_lang)),
            ]
        )
        try:
            results = await asyncio.to_thread(
                self._client.search,
                collection_name=self._collection,
                query_vector=query_vector.tolist(),
                query_filter=flt,
                limit=top_k,
            )
        except Exception as exc:
            raise RetrievalError(f"Qdrant search failed: {exc}") from exc
        return [
            {
                "entry": {
                    "source_text": r.payload["source_text"],
                    "target_text": r.payload["target_text"],
                },
                "score": float(r.score),
                "source": "vector",
            }
            for r in results
        ]

    async def delete_by_filter(self, source_lang: str, target_lang: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        selector = FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="source_lang", match=MatchValue(value=source_lang)
                    ),
                    FieldCondition(
                        key="target_lang", match=MatchValue(value=target_lang)
                    ),
                ]
            )
        )
        await asyncio.to_thread(
            self._client.delete, collection_name=self._collection, points_selector=selector
        )

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
