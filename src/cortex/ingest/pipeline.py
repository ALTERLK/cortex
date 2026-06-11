"""End-to-end ingest pipeline: files on disk → vectors in Qdrant.

Usage::

    from cortex.ingest.pipeline import ingest_directory
    from cortex.ingest.embedder import Embedder
    from cortex.ingest.store import VectorStore

    store = VectorStore()
    embedder = Embedder()
    store.ensure_collection(embedder.dimension)
    n = ingest_directory(Path("my_docs"), store, embedder)
    print(f"Ingested {n} chunks.")

NOTE (learning): this function is intentionally *not* async and *not* batched
across files — clarity over performance at this stage. When the corpus grows
to thousands of files, batching embed() calls across chunks from multiple
files would be the first optimization.
"""

from __future__ import annotations

from pathlib import Path

from cortex.ingest.chunker import RecursiveCharacterChunker
from cortex.ingest.embedder import Embedder
from cortex.ingest.loader import load_document
from cortex.ingest.store import VectorStore


_SUPPORTED_GLOB = "**/*.{md,txt,pdf}"


def ingest_directory(
    directory: Path,
    store: VectorStore,
    embedder: Embedder,
    *,
    chunk_size: int = 400,
    overlap: int = 50,
    verbose: bool = True,
) -> int:
    """Load, chunk, embed, and upsert all documents in *directory*.

    Args:
        directory:  Root folder to walk (recursively).
        store:      Pre-initialised VectorStore with an existing collection.
        embedder:   Pre-initialised Embedder.
        chunk_size: Characters per chunk.
        overlap:    Overlap between consecutive chunks.
        verbose:    Print one line per file if True.

    Returns:
        Total number of chunks ingested.
    """
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, overlap=overlap)
    total = 0

    patterns = ["**/*.md", "**/*.txt", "**/*.pdf"]
    paths = sorted(
        {p for pattern in patterns for p in directory.glob(pattern)}
    )

    for path in paths:
        try:
            doc = load_document(path)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {path.name}: {exc}")
            continue

        chunks = chunker.chunk(doc)
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        store.upsert(chunks, embeddings)
        total += len(chunks)

        if verbose:
            print(f"  [ok] {path.name}: {len(chunks)} chunks")

    return total
