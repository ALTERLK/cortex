"""Dense retriever: embed a query and fetch the nearest chunks from the store.

Deliberately thin — embedding lives in `Embedder`, ranking lives in Qdrant.
"""

from __future__ import annotations

from dataclasses import replace

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import RetrievalTrace, SearchResult, VectorStore


class Retriever:
    """Semantic search over the vector store.

    Args:
        store:    Populated VectorStore instance.
        embedder: Embedder used at ingest time. Queries must be embedded with
                  the same model, or the vectors live in different spaces and
                  similarity scores are meaningless.
        top_k:    Number of passages to return per query.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the top-k most similar passages for *query*."""
        vector = self._embedder.embed([query])[0]
        hits = self._store.search(vector, top_k=top_k or self._top_k)
        return [
            replace(hit, trace=RetrievalTrace(dense_rank=rank))
            for rank, hit in enumerate(hits, 1)
        ]
