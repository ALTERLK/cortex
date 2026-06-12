"""Tests for the vector store and ingest pipeline.

All tests run offline:
- Qdrant in ":memory:" mode — no disk I/O, no Docker.
- FakeEmbedder produces random-but-dimensionally-correct vectors so we
  never load the real sentence-transformers model.
"""

from __future__ import annotations

import random

from cortex.ingest.chunker import Chunk
from cortex.ingest.store import VectorStore, _stable_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 16  # tiny dimension — enough to test logic, fast to compute


def fake_vector(dim: int = DIM) -> list[float]:
    """Unit-normalised random vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def make_chunk(source: str = "test.md", idx: int = 0, text: str = "hello") -> Chunk:
    return Chunk(text=text, metadata={"source": source, "chunk_index": idx})


def fresh_store() -> VectorStore:
    store = VectorStore(":memory:")
    store.ensure_collection(DIM)
    return store


# ---------------------------------------------------------------------------
# _stable_id
# ---------------------------------------------------------------------------


def test_stable_id_is_deterministic() -> None:
    assert _stable_id("notes.md", 3) == _stable_id("notes.md", 3)


def test_stable_id_differs_for_different_inputs() -> None:
    assert _stable_id("a.md", 0) != _stable_id("b.md", 0)
    assert _stable_id("a.md", 0) != _stable_id("a.md", 1)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


def test_empty_store_has_zero_count() -> None:
    store = fresh_store()
    assert store.count() == 0


def test_upsert_increases_count() -> None:
    store = fresh_store()
    chunks = [make_chunk(idx=i) for i in range(3)]
    vectors = [fake_vector() for _ in chunks]
    store.upsert(chunks, vectors)
    assert store.count() == 3


def test_upsert_is_idempotent() -> None:
    store = fresh_store()
    chunks = [make_chunk(idx=0, text="original")]
    store.upsert(chunks, [fake_vector()])
    assert store.count() == 1
    # Re-upsert same logical chunk — count must stay at 1.
    store.upsert(chunks, [fake_vector()])
    assert store.count() == 1


def test_search_returns_top_k_results() -> None:
    store = fresh_store()
    chunks = [make_chunk(source="doc.md", idx=i, text=f"chunk {i}") for i in range(10)]
    vectors = [fake_vector() for _ in chunks]
    store.upsert(chunks, vectors)

    query = fake_vector()
    results = store.search(query, top_k=3)
    assert len(results) == 3


def test_search_result_fields() -> None:
    store = fresh_store()
    chunk = make_chunk(source="notes.md", idx=7, text="some content")
    store.upsert([chunk], [fake_vector()])

    results = store.search(fake_vector(), top_k=1)
    assert len(results) == 1
    r = results[0]
    assert r.text == "some content"
    assert r.source == "notes.md"
    assert r.chunk_index == 7
    assert -1.0 <= r.score <= 1.0


def test_search_scores_are_ordered_descending() -> None:
    store = fresh_store()
    chunks = [make_chunk(idx=i) for i in range(5)]
    store.upsert(chunks, [fake_vector() for _ in chunks])

    results = store.search(fake_vector(), top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# list_sources / get_file_hash / delete_source
# ---------------------------------------------------------------------------


def test_list_sources_distinct_and_sorted() -> None:
    store = fresh_store()
    chunks = [make_chunk("b.md", 0), make_chunk("a.md", 0), make_chunk("a.md", 1)]
    store.upsert(chunks, [fake_vector() for _ in chunks])
    assert store.list_sources() == ["a.md", "b.md"]


def test_list_sources_empty_store() -> None:
    assert fresh_store().list_sources() == []


def test_get_file_hash_roundtrip() -> None:
    store = fresh_store()
    chunk = Chunk(text="x", metadata={"source": "doc.md", "chunk_index": 0, "file_hash": "abc123"})
    store.upsert([chunk], [fake_vector()])
    assert store.get_file_hash("doc.md") == "abc123"
    assert store.get_file_hash("missing.md") is None


def test_delete_source_removes_only_that_source() -> None:
    store = fresh_store()
    chunks = [make_chunk("keep.md", 0), make_chunk("drop.md", 0), make_chunk("drop.md", 1)]
    store.upsert(chunks, [fake_vector() for _ in chunks])
    store.delete_source("drop.md")
    assert store.count() == 1
    assert store.list_sources() == ["keep.md"]


# ---------------------------------------------------------------------------
# Incremental ingest (pipeline-level)
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    """Fake embedder that counts how many texts it was asked to embed."""

    dimension = DIM

    def __init__(self) -> None:
        self.embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [fake_vector() for _ in texts]


def test_ingest_skips_unchanged_files(tmp_path) -> None:
    from cortex.ingest.pipeline import ingest_directory

    (tmp_path / "a.md").write_text("Paragraph one.\n\nParagraph two.", encoding="utf-8")
    store = fresh_store()
    embedder = _CountingEmbedder()

    first = ingest_directory(tmp_path, store, embedder, verbose=False)
    assert first >= 1
    first_embedded = embedder.embedded

    # Second run: nothing changed -> nothing re-embedded, nothing written.
    second = ingest_directory(tmp_path, store, embedder, verbose=False)
    assert second == 0
    assert embedder.embedded == first_embedded


def test_ingest_changed_file_leaves_no_stale_chunks(tmp_path) -> None:
    from cortex.ingest.pipeline import ingest_directory

    # Long file -> several chunks.
    (tmp_path / "a.md").write_text("\n\n".join(f"Paragraph {i}. " + "x" * 300 for i in range(6)))
    store = fresh_store()
    embedder = _CountingEmbedder()
    ingest_directory(tmp_path, store, embedder, verbose=False)
    big_count = store.count()

    # Shrink the file -> re-ingest must not leave the old tail chunks behind.
    (tmp_path / "a.md").write_text("Tiny now.")
    ingest_directory(tmp_path, store, embedder, verbose=False)
    assert store.count() < big_count
    assert store.count() == 1


def test_ingest_uses_relative_path_as_source(tmp_path) -> None:
    from cortex.ingest.pipeline import ingest_directory

    (tmp_path / "en").mkdir()
    (tmp_path / "zh").mkdir()
    (tmp_path / "en" / "index.md").write_text("English content here.")
    (tmp_path / "zh" / "index.md").write_text("Chinese content here.")
    store = fresh_store()
    ingest_directory(tmp_path, store, _CountingEmbedder(), verbose=False)
    assert store.list_sources() == ["en/index.md", "zh/index.md"]
