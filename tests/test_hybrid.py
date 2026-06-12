"""Tests for the BM25 index and the hybrid (RRF) retriever — all offline."""

from __future__ import annotations

from cortex.ingest.store import SearchResult
from cortex.rag.bm25 import BM25Index, tokenize
from cortex.rag.hybrid import HybridRetriever

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_english_lowercases() -> None:
    assert tokenize("Hello FastAPI World") == ["hello", "fastapi", "world"]


def test_tokenize_strips_punctuation() -> None:
    assert tokenize("uv-cache: clean! (now)") == ["uv", "cache", "clean", "now"]


def test_tokenize_chinese_segments() -> None:
    tokens = tokenize("查询参数的默认值")
    assert len(tokens) >= 2  # jieba splits the phrase into words
    assert all(tok.strip() for tok in tokens)


def test_tokenize_mixed_chinese_english() -> None:
    tokens = tokenize("FastAPI 的查询参数")
    assert "fastapi" in tokens
    assert any("参数" in tok for tok in tokens)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

DOCS = [
    "FastAPI uses CORSMiddleware to handle cross origin requests.",
    "uv cache clean removes cached packages from the uv cache.",
    "Background tasks run after the response is sent to the client.",
    "Query parameters are function parameters not part of the path.",
]


def make_index() -> BM25Index:
    index = BM25Index()
    index.fit(DOCS)
    return index


def test_bm25_exact_term_ranks_first() -> None:
    hits = make_index().search("CORSMiddleware cross origin")
    assert hits[0].doc_id == 0


def test_bm25_rare_term_beats_common_term() -> None:
    # "uv" appears twice in doc 1 only; "the" appears in several docs.
    hits = make_index().search("uv cache")
    assert hits[0].doc_id == 1


def test_bm25_no_match_returns_empty() -> None:
    assert make_index().search("kubernetes deployment replicas") == []


def test_bm25_empty_index_returns_empty() -> None:
    assert BM25Index().search("anything") == []


def test_bm25_scores_descending() -> None:
    hits = make_index().search("parameters path response")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# HybridRetriever (RRF fusion)
# ---------------------------------------------------------------------------


def chunk(i: int, text: str) -> SearchResult:
    return SearchResult(score=0.0, text=text, source=f"doc{i}.md", chunk_index=0)


class _FakeStore:
    """Store whose dense search returns a fixed ranking."""

    def __init__(self, chunks: list[SearchResult], dense_order: list[int]) -> None:
        self._chunks = chunks
        self._dense_order = dense_order

    def all_chunks(self) -> list[SearchResult]:
        return self._chunks

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        return [self._chunks[i] for i in self._dense_order[:top_k]]


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


def test_hybrid_fuses_dense_and_sparse() -> None:
    chunks = [chunk(i, text) for i, text in enumerate(DOCS)]
    # Dense thinks doc3 then doc2; BM25 will rank doc1 first for "uv cache".
    store = _FakeStore(chunks, dense_order=[3, 2, 0, 1])
    retriever = HybridRetriever(store, _FakeEmbedder(), top_k=3)

    results = retriever.retrieve("uv cache")
    sources = [r.source for r in results]
    # doc1 is ranked #1 by sparse AND #4 by dense -> strong fused position;
    # doc3 is dense #1 only. Both must appear in the top results.
    assert "doc1.md" in sources
    assert "doc3.md" in sources


def test_hybrid_doc_in_both_lists_beats_single_list() -> None:
    chunks = [chunk(i, text) for i, text in enumerate(DOCS)]
    # doc1 is #2 dense AND #1 sparse for "uv cache" -> must beat doc3 (#1 dense only).
    store = _FakeStore(chunks, dense_order=[3, 1, 2, 0])
    retriever = HybridRetriever(store, _FakeEmbedder(), top_k=2)

    results = retriever.retrieve("uv cache")
    assert results[0].source == "doc1.md"


def test_hybrid_respects_top_k_override() -> None:
    chunks = [chunk(i, text) for i, text in enumerate(DOCS)]
    retriever = HybridRetriever(_FakeStore(chunks, [0, 1, 2, 3]), _FakeEmbedder(), top_k=4)
    assert len(retriever.retrieve("parameters", top_k=2)) == 2


# ---------------------------------------------------------------------------
# RerankingRetriever (two-stage)
# ---------------------------------------------------------------------------


class _ReverseReranker:
    """Fake cross-encoder: scores candidates in reverse input order."""

    def rerank(self, query, results, top_k=5):
        return list(reversed(results))[:top_k]


class _ListRetriever:
    def __init__(self, results):
        self._results = results
        self.last_top_k = None

    def retrieve(self, query, top_k=None):
        self.last_top_k = top_k
        return self._results[:top_k]


def test_reranking_retriever_fetches_candidates_then_truncates() -> None:
    from cortex.rag.reranker import RerankingRetriever

    candidates = [chunk(i, f"text {i}") for i in range(20)]
    base = _ListRetriever(candidates)
    retriever = RerankingRetriever(base, _ReverseReranker(), top_k=5, candidate_k=20)

    results = retriever.retrieve("q")
    assert base.last_top_k == 20          # wide candidate fetch
    assert len(results) == 5              # narrow final cut
    assert results[0].source == "doc19.md"  # reranker reordered
