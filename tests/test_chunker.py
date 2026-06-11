"""Unit tests for the document loader and chunker.

All tests are offline — no file I/O beyond a tiny temp file for the loader,
no network calls.
"""

import pytest

from cortex.ingest.chunker import RecursiveCharacterChunker
from cortex.ingest.loader import Document, load_document

# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


def make_doc(text: str, source: str = "test.md") -> Document:
    return Document(text=text, source=source)


def test_empty_document_returns_no_chunks() -> None:
    chunks = RecursiveCharacterChunker().chunk(make_doc(""))
    assert chunks == []


def test_whitespace_only_document_returns_no_chunks() -> None:
    chunks = RecursiveCharacterChunker().chunk(make_doc("   \n\n   "))
    assert chunks == []


def test_short_document_is_a_single_chunk() -> None:
    text = "This is a short document."
    chunks = RecursiveCharacterChunker(chunk_size=400).chunk(make_doc(text))
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_all_chunks_fit_within_chunk_size() -> None:
    text = " ".join(f"word{i}" for i in range(300))  # ~1800 chars
    chunks = RecursiveCharacterChunker(chunk_size=100, overlap=20).chunk(
        make_doc(text)
    )
    assert chunks  # at least one chunk
    for c in chunks:
        assert len(c.text) <= 100, f"chunk too long: {len(c.text)}"


def test_metadata_source_and_index() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = RecursiveCharacterChunker(chunk_size=200, overlap=0).chunk(
        make_doc(text, source="notes.md")
    )
    assert all(c.metadata["source"] == "notes.md" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_more_chunks_with_smaller_chunk_size() -> None:
    text = "word " * 200  # 1000 chars
    small = RecursiveCharacterChunker(chunk_size=80, overlap=0).chunk(make_doc(text))
    large = RecursiveCharacterChunker(chunk_size=300, overlap=0).chunk(make_doc(text))
    assert len(small) > len(large)


def test_overlap_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RecursiveCharacterChunker(chunk_size=100, overlap=100)


def test_consecutive_chunks_share_overlap_content() -> None:
    # Paragraphs are ~26 chars each — short enough to fit in the overlap
    # budget (40), so the chunker will carry the last paragraph of chunk[0]
    # into the start of chunk[1].
    paragraphs = [f"Para {i}: some content here." for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = RecursiveCharacterChunker(chunk_size=80, overlap=40).chunk(
        make_doc(text)
    )
    assert len(chunks) >= 2
    # The last paragraph of chunk[0] must appear at the start of chunk[1].
    last_line_of_first = chunks[0].text.split("\n")[-1]
    assert last_line_of_first in chunks[1].text


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_markdown_file(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "note.md"
    p.write_text("# Hello\n\nWorld.", encoding="utf-8")
    doc = load_document(p)
    assert doc.source == "note.md"
    assert "Hello" in doc.text
    assert "World" in doc.text


def test_load_txt_file(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "plain.txt"
    p.write_text("line one\nline two", encoding="utf-8")
    doc = load_document(p)
    assert doc.source == "plain.txt"
    assert "line one" in doc.text


def test_load_unsupported_extension_raises(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(p)


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_document("does_not_exist.md")
