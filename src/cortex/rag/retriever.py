"""Retriever: embed a query and fetch the most relevant chunks from the store.

NOTE (learning): the Retriever is deliberately thin — it only converts a
natural-language query into a vector and delegates to the VectorStore. All
ranking logic lives in Qdrant; all embedding logic lives in Embedder. Each
class has one job.
"""

from __future__ import annotations

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import SearchResult, VectorStore


class Retriever:
    """Semantic search over the vector store.

    Args:
        store:   Populated VectorStore instance.
        embedder: Embedder used at ingest time (must be the same model).
        top_k:   Number of passages to return per query.
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
        """Return the top-k most relevant passages for *query*.

        Args:
            query: Natural-language search query.
            top_k: Optional per-call override of the instance default.

        NOTE (learning): the query is embedded with the same model used at
        ingest time. Using a different model would compare apples to oranges —
        the vectors would be in different spaces and similarity scores would be
        meaningless.
        """
        vector = self._embedder.embed([query])[0]
        return self._store.search(vector, top_k=top_k or self._top_k)
