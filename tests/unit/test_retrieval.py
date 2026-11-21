"""Юнит-тесты retrieval: BM25, RRF-слияние, HybridRetriever, point_id."""

from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever, rrf_merge
from backend.retrieval.vector_index import point_id


def _entries() -> list[dict[str, str]]:
    return [
        {"source_text": "осуществить оплату в установленный срок", "target_text": "оплатить вовремя"},
        {"source_text": "направить уведомление заявителю", "target_text": "сообщить заявителю"},
        {"source_text": "произвести возврат денежных средств", "target_text": "вернуть деньги"},
    ]


# ---------- BM25 ----------


def test_bm25_empty_index_returns_empty() -> None:
    assert BM25Index().search("оплата", "bureau", "plain") == []


def test_bm25_search_top_k_and_rebuild() -> None:
    index = BM25Index()
    index.add_entries(_entries(), "bureau", "plain")
    results = index.search("оплату срок", "bureau", "plain", top_k=2)
    assert 1 <= len(results) <= 2
    assert all(r["source"] == "bm25" and r["score"] > 0 for r in results)
    assert results[0]["entry"]["source_text"].startswith("осуществить")

    # ленивая пересборка после дозаписи
    index.add_entries(
        [{"source_text": "срок исковой давности", "target_text": "время на иск"}],
        "bureau", "plain",
    )
    # «срок» есть в 2 из 4 документов -> idf=0 у rank_bm25, поэтому ищем
    # дискриминативный термин новой записи
    again = index.search("исковой давности", "bureau", "plain", top_k=5)
    assert any("исковой" in r["entry"]["source_text"] for r in again)


def test_bm25_pair_isolation_and_clear() -> None:
    index = BM25Index()
    index.add_entries(_entries(), "bureau", "plain")
    assert index.search("оплата", "legal", "plain") == []
    index.clear()
    assert index.search("оплата", "bureau", "plain") == []


# ---------- RRF ----------


def test_rrf_dedup_and_boost() -> None:
    shared = {"source_text": "A", "target_text": "a"}
    only_bm25 = {"source_text": "B", "target_text": "b"}
    bm25 = [
        {"entry": shared, "score": 5.0, "source": "bm25"},
        {"entry": only_bm25, "score": 1.0, "source": "bm25"},
    ]
    vector = [{"entry": shared, "score": 0.9, "source": "vector"}]
    merged = rrf_merge(bm25, vector, "bureau", "plain", top_k=5)
    assert len(merged) == 2  # дубликат слит
    assert merged[0]["entry"] is shared
    # score общей записи = 1/61 + 1/61
    assert abs(merged[0]["score"] - 2 / 61) < 1e-9
    assert merged[0]["source"] == "hybrid"


def test_rrf_top_k() -> None:
    bm25 = [
        {"entry": {"source_text": f"S{i}", "target_text": f"t{i}"}, "score": 1.0}
        for i in range(10)
    ]
    merged = rrf_merge(bm25, [], "bureau", "plain", top_k=3)
    assert len(merged) == 3


# ---------- Hybrid ----------


class _FakeVector:
    def __init__(self, results):
        self._results = results

    async def search(self, vec, source_lang, target_lang, top_k=5):
        return self._results[:top_k]


class _FakeEmbeddings:
    async def encode_async(self, texts):
        return [[0.0] for _ in texts]


async def test_hybrid_bm25_only_when_no_vector() -> None:
    bm25 = BM25Index()
    bm25.add_entries(_entries(), "bureau", "plain")
    retriever = HybridRetriever(bm25, None, _FakeEmbeddings())
    results = await retriever.search("оплату", "bureau", "plain")
    assert results
    assert all(r["source"] == "hybrid" for r in results)


async def test_hybrid_merges_vector_results() -> None:
    bm25 = BM25Index()
    bm25.add_entries(_entries(), "bureau", "plain")
    extra = {"source_text": "векторная находка", "target_text": "простое"}
    vector = _FakeVector([{"entry": extra, "score": 0.95, "source": "vector"}])
    retriever = HybridRetriever(bm25, vector, _FakeEmbeddings())
    results = await retriever.search("оплату", "bureau", "plain", top_k=5)
    sources = {r["entry"]["source_text"] for r in results}
    assert "векторная находка" in sources


# ---------- point_id ----------


def test_point_id_deterministic_and_unique() -> None:
    a = point_id("bureau", "plain", "текст", "простой")
    b = point_id("bureau", "plain", "текст", "простой")
    c = point_id("bureau", "plain", "другой", "простой")
    assert a == b
    assert a != c
