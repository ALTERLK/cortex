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

import hashlib
from pathlib import Path

from cortex.ingest.chunker import RecursiveCharacterChunker
from cortex.ingest.embedder import Embedder
from cortex.ingest.loader import load_document
from cortex.ingest.store import VectorStore


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

    Incremental: each file's content hash is stored with its chunks; on
    re-runs, unchanged files are skipped entirely (no re-embedding). When a
    file HAS changed, its old chunks are deleted before the new ones are
    written, so a file that shrinks never leaves stale chunks behind.

    Args:
        directory:  Root folder to walk (recursively).
        store:      Pre-initialised VectorStore with an existing collection.
        embedder:   Pre-initialised Embedder.
        chunk_size: Characters per chunk.
        overlap:    Overlap between consecutive chunks.
        verbose:    Print one line per file if True.

    Returns:
        Total number of chunks ingested (skipped files contribute 0).
    """
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, overlap=overlap)
    total = 0
    skipped = 0

    patterns = ["**/*.md", "**/*.txt", "**/*.pdf"]
    paths = sorted({p for pattern in patterns for p in directory.glob(pattern)})

    for path in paths:
        # Source label = path relative to the ingest root, so files with the
        # same name in different folders (en/index.md vs zh/index.md) stay
        # distinct in citations and in the index.
        source = path.relative_to(directory).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if store.get_file_hash(source) == file_hash:
            skipped += 1
            if verbose:
                print(f"  [unchanged] {source}")
            continue

        try:
            doc = load_document(path)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {source}: {exc}")
            continue

        doc.source = source
        chunks = chunker.chunk(doc)
        if not chunks:
            continue

        for c in chunks:
            c.metadata["file_hash"] = file_hash

        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        store.delete_source(source)  # changed file: remove old chunks first
        store.upsert(chunks, embeddings)
        total += len(chunks)

        if verbose:
            print(f"  [ok] {source}: {len(chunks)} chunks")

    if verbose and skipped:
        print(f"  ({skipped} unchanged files skipped)")
    return total
