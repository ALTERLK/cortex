"""Document loader: read files from disk into plain text + metadata.

Supported formats: .md, .txt (any UTF-8 text file), .pdf.

NOTE (learning): the loader's only job is "bytes on disk → Python string."
It does NOT chunk or embed — each stage of the pipeline does exactly one
thing, so each stage can be tested and swapped independently.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A raw document as loaded from disk.

    Attributes:
        text:     Full plain-text content.
        source:   Filename (no directory path) used as a metadata label.
        metadata: Any extra key/value pairs (e.g. page_count for PDFs).
    """

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_document(path: Path | str) -> Document:
    """Load a single file and return a Document.

    Args:
        path: Path to a .md, .txt, or .pdf file.

    Returns:
        Document with full text and source filename.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in {".md", ".txt", ""}:
        return _load_text(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: .md, .txt, .pdf"
        )


def _load_text(path: Path) -> Document:
    return Document(
        text=path.read_text(encoding="utf-8"),
        source=path.name,
    )


def _load_pdf(path: Path) -> Document:
    # NOTE (learning): pypdf is a pure-Python PDF reader. We join all pages
    # with a double newline so paragraph-level chunking treats page breaks
    # the same way it treats paragraph breaks.
    from pypdf import PdfReader  # imported lazily: only needed for PDF paths

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    return Document(
        text=text,
        source=path.name,
        metadata={"page_count": len(reader.pages)},
    )
