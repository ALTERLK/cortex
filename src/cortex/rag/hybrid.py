"""Hybrid retriever: dense (BGE-M3 / Qdrant) + sparse (BM25) fused with RRF.

Why hybrid? Dense embeddings capture meaning ("how do I allow another
origin?" finds the CORS page) but are weak on exact rare tokens (error
codes, flag names, `--refresh`). BM25 is the mirror image. Fusing both
covers each one's blind spot.

Reciprocal Rank Fusion combines the two ranked lists using only RANKS:

    rrf(d) = sum over lists L of  1 / (k + rank_L(d))      (k = 60)

Cosine similarities and BM25 weights live on incomparable scales; ranks are
always comparable. k = 60 (the constant from the original RRF paper) dampens
the head so one list's #1 cannot dominate the fusion.
"""

from __future__ import annotations

from dataclasses import replace

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import RetrievalTrace, SearchResult, VectorStore
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
        self._chunks: list[SearchResult] = []
        self._bm25 = BM25Index()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the sparse index from the chunks currently in the store.

        Qdrant is the single source of truth; BM25 is a derived in-memory
        index, so it must be rebuilt whenever documents are added or removed.
        The new index is built on the side and swapped in with one
        assignment, so a query running concurrently sees either the old
        index or the new one — never a half-built mix of chunks and scores.
        """
        chunks = self._store.all_chunks()
        bm25 = BM25Index()
        bm25.fit([c.text for c in chunks])
        self._chunks, self._bm25 = chunks, bm25

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the fused top-k passages for *query*."""
        k = top_k or self._top_k
        chunks, bm25 = self._chunks, self._bm25  # one consistent snapshot

        dense = self._store.search(
            self._embedder.embed([query])[0], top_k=self._candidate_k
        )
        sparse = [chunks[hit.doc_id] for hit in bm25.search(query, top_k=self._candidate_k)]

        fused: dict[tuple[str, int], float] = {}
        by_key: dict[tuple[str, int], SearchResult] = {}
        ranks: dict[tuple[str, int], dict[str, int]] = {}
        for list_name, ranked_list in (("dense_rank", dense), ("sparse_rank", sparse)):
            for rank, result in enumerate(ranked_list, 1):
                key = (result.source, result.chunk_index)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                by_key.setdefault(key, result)
                ranks.setdefault(key, {})[list_name] = rank

        top = sorted(fused, key=lambda key: fused[key], reverse=True)[:k]
        # The reported score is the RRF score — comparable within one
        # response, not across retrieval strategies.
        return [
            replace(
                by_key[key],
                score=round(fused[key], 6),
                trace=RetrievalTrace(rrf_score=round(fused[key], 6), **ranks[key]),
            )
            for key in top
        ]
