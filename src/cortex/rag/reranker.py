"""Cross-encoder reranker: re-score retrieved candidates with a stronger model.

A bi-encoder (BGE-M3) embeds query and document separately, so similarity is
a cheap dot product over precomputed vectors — fast enough to search the
whole corpus. A cross-encoder reads the query and the document together
through one transformer: far more accurate, far too slow to run on every
chunk. Hence two stages: the cheap retriever fetches ~20 candidates, the
cross-encoder reorders them, and the top 5 go to the LLM.
"""

from __future__ import annotations

from dataclasses import replace

from cortex.ingest.store import RetrievalTrace, SearchResult


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
        return [
            replace(
                results[i],
                score=round(float(scores[i]), 4),
                # Keep the upstream ranks and add this stage's score to them.
                trace=replace(
                    results[i].trace or RetrievalTrace(),
                    rerank_score=round(float(scores[i]), 4),
                ),
            )
            for i in order[:top_k]
        ]


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

    def refresh(self) -> None:
        """Propagate an index refresh to the base retriever, if it has one."""
        refresh = getattr(self._base, "refresh", None)
        if refresh is not None:
            refresh()
