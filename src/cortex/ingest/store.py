"""Qdrant vector store wrapper.

Handles collection management, idempotent upserts, and similarity search.
All Qdrant SDK calls are confined to this file — the rest of the codebase
sees only plain Python dicts.

NOTE (learning): we use Qdrant's *local file* mode (QdrantClient(path=...)),
which stores the index on disk without requiring a running Docker container.
The API is identical to the Docker/cloud modes — swapping later is a one-line
config change.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cortex.ingest.chunker import Chunk


COLLECTION_NAME = "cortex"


@dataclass
class SearchResult:
    """A single hit returned by the vector store."""

    score: float      # cosine similarity in [−1, 1]; higher is more similar
    text: str         # chunk text
    source: str       # origin filename
    chunk_index: int  # position within that file


class VectorStore:
    """Thin wrapper around a Qdrant collection.

    Args:
        path: Directory where Qdrant stores its on-disk index.
              Pass ``":memory:"`` for an in-memory store (useful in tests).
    """

    def __init__(self, path: str = "data/qdrant") -> None:
        from qdrant_client import QdrantClient

        if path == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=path)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunk vectors.

        Each point ID is a deterministic UUID derived from (source, chunk_index),
        so re-ingesting the same file overwrites existing vectors instead of
        creating duplicates.

        NOTE (learning): this "idempotent write" pattern is essential for any
        pipeline that might re-run on the same data. Using a content-derived ID
        means the same logical chunk always maps to the same storage slot.
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_stable_id(c.metadata["source"], c.metadata["chunk_index"]),
                vector=emb,
                payload={"text": c.text, **c.metadata},
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the top-k most similar chunks for a query vector."""
        # NOTE (learning): qdrant-client >=1.7 replaced client.search() with
        # client.query_points(); the result is a QueryResponse with a .points list.
        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        )
        return [
            SearchResult(
                score=h.score,
                text=h.payload.get("text", ""),
                source=h.payload.get("source", ""),
                chunk_index=h.payload.get("chunk_index", -1),
            )
            for h in response.points
        ]

    def count(self) -> int:
        """Return the total number of stored vectors."""
        return self._client.count(collection_name=COLLECTION_NAME).count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(source: str, chunk_index: int) -> str:
    """Deterministic UUID from (source, chunk_index)."""
    key = f"{source}:{chunk_index}".encode()
    digest = hashlib.sha256(key).digest()
    return str(uuid.UUID(bytes=digest[:16]))
