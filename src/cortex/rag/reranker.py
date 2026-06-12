"""Cross-encoder reranker: re-score retrieved candidates with a stronger model.

NOTE (learning): bi-encoders (BGE-M3) embed query and document SEPARATELY,
so similarity is a cheap dot product over precomputed vectors — that's what
makes search over thousands of chunks fast. A cross-encoder reads the query
and the document TOGETHER through one transformer, which is far more
accurate but far too slow to run against the whole corpus. The standard
two-stage design: cheap retriever fetches ~20 candidates, expensive
cross-encoder reorders them, top-5 go to the LLM.
"""

from __future__ import annotations

from dataclasses import replace

from cortex.ingest.store import SearchResult


class Reranker:
    """bge-reranker-v2-m3 cross-encoder (local, multilingual, ~1.1 GB).

    Args:
        model_name: HuggingFace cross-encoder model id.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Lazy heavy import, same pattern as Embedder.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Re-score *results* against *query*; return the top_k best."""
        if not results:
            return []
        scores = self._model.predict([(query, r.text) for r in results])
        order = sorted(range(len(results)), key=lambda i: float(scores[i]), reverse=True)
        return [replace(results[i], score=round(float(scores[i]), 4)) for i in order[:top_k]]


class RerankingRetriever:
    """Two-stage retriever: any base retriever + cross-encoder rerank.

    Drop-in replacement for Retriever/HybridRetriever (same retrieve()
    signature) so eval and the API can swap strategies via config.

    Args:
        base:        Retriever or HybridRetriever supplying candidates.
        reranker:    Cross-encoder used for the second stage.
        top_k:       Results returned per query.
        candidate_k: Candidates fetched from the base before reranking.
    """

    def __init__(self, base: object, reranker: Reranker, top_k: int = 5, candidate_k: int = 20) -> None:
        self._base = base
        self._reranker = reranker
        self._top_k = top_k
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        candidates = self._base.retrieve(query, top_k=self._candidate_k)
        return self._reranker.rerank(query, candidates, top_k=top_k or self._top_k)
