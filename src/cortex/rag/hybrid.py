"""Hybrid retriever: dense (BGE-M3 / Qdrant) + sparse (BM25) fused with RRF.

Why hybrid? Dense embeddings capture meaning ("how do I allow another
origin?" finds the CORS page) but are weak on exact rare tokens (error
codes, flag names, `--refresh`). BM25 is the mirror image. Fusing both
covers each one's blind spot.

NOTE (learning): RRF (Reciprocal Rank Fusion) combines ranked lists using
only the RANKS, not the raw scores:

    rrf(d) = sum over lists L of  1 / (k + rank_L(d))      (k = 60)

Raw scores from different systems live on incomparable scales (cosine
similarity vs BM25 weights); ranks are always comparable. k=60 dampens the
head so one list's #1 can't dominate everything — the standard constant
from the original RRF paper.
"""

from __future__ import annotations

from dataclasses import replace

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import SearchResult, VectorStore
from cortex.rag.bm25 import BM25Index


class HybridRetriever:
    """Dense + BM25 retrieval with Reciprocal Rank Fusion.

    Drop-in replacement for Retriever (same retrieve() signature).

    Args:
        store:       Populated VectorStore.
        embedder:    Same embedding model used at ingest time.
        top_k:       Results returned per query.
        candidate_k: How many candidates EACH retriever contributes before
                     fusion. Larger = better recall, slower fusion.
        rrf_k:       RRF dampening constant.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

        # Build the sparse index once from the chunks already in Qdrant —
        # one source of truth, nothing extra to keep in sync on re-ingest
        # (the API server builds this at startup).
        self._chunks = store.all_chunks()
        self._bm25 = BM25Index()
        self._bm25.fit([c.text for c in self._chunks])

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the fused top-k passages for *query*."""
        k = top_k or self._top_k

        dense = self._store.search(
            self._embedder.embed([query])[0], top_k=self._candidate_k
        )
        sparse = [
            self._chunks[hit.doc_id]
            for hit in self._bm25.search(query, top_k=self._candidate_k)
        ]

        fused: dict[tuple[str, int], float] = {}
        by_key: dict[tuple[str, int], SearchResult] = {}
        for ranked_list in (dense, sparse):
            for rank, result in enumerate(ranked_list, 1):
                key = (result.source, result.chunk_index)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                by_key.setdefault(key, result)

        top = sorted(fused, key=lambda key: fused[key], reverse=True)[:k]
        # Report the RRF score — comparable within one response, not across systems.
        return [replace(by_key[key], score=round(fused[key], 6)) for key in top]
