"""Recursive character chunker.

Splits a Document into overlapping Chunks that each fit within `chunk_size`
characters. The strategy:

  1. _split_text: try separators in priority order (\n\n → \n → space → char)
     until every piece is ≤ chunk_size.
  2. _merge_with_overlap: greedily pack pieces into chunks up to chunk_size,
     then carry back `overlap` characters into the next chunk's starting set.

NOTE (learning): overlap exists because a complete sentence might straddle a
chunk boundary. By repeating the tail of the previous chunk at the head of the
next one, we ensure at least one chunk contains the full sentence.
"""

from dataclasses import dataclass, field
from typing import Any

from cortex.ingest.loader import Document

# Default separator hierarchy: paragraph → line → word → character.
_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


@dataclass
class Chunk:
    """A single text window extracted from a Document.

    Attributes:
        text:     The chunk's plain-text content.
        metadata: Provenance info: at minimum ``source`` and ``chunk_index``.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RecursiveCharacterChunker:
    """Split documents into overlapping fixed-size chunks.

    Args:
        chunk_size:  Maximum number of characters per chunk (default 400).
        overlap:     Characters of overlap between consecutive chunks (default 50).
        separators:  Separator hierarchy to try when splitting (default:
                     paragraph → line → word → character).
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else _DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into a list of Chunks with source metadata."""
        if not doc.text.strip():
            return []
        pieces = self._split_text(doc.text, self.separators)
        texts = self._merge_with_overlap(pieces)
        return [
            Chunk(
                text=t,
                metadata={"source": doc.source, "chunk_index": i},
            )
            for i, t in enumerate(texts)
        ]

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* until every piece fits in chunk_size."""
        sep, *rest = separators

        # Base case: character-level split (sep == "").
        if sep == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        pieces: list[str] = []
        for raw in text.split(sep):
            stripped = raw.strip()
            if not stripped:
                continue
            if len(stripped) <= self.chunk_size:
                pieces.append(stripped)
            elif rest:
                # NOTE (learning): this is the "recursive" part — we fall back
                # to a finer separator only when the current one leaves pieces
                # that are still too large.
                pieces.extend(self._split_text(stripped, rest))
            else:
                # No separators left; force-split by character.
                pieces.extend(
                    stripped[i : i + self.chunk_size]
                    for i in range(0, len(stripped), self.chunk_size)
                )
        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily pack *pieces* into chunks of ≤ chunk_size with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        cur_len = 0  # length of "\n".join(current)

        for piece in pieces:
            # +1 accounts for the '\n' separator added when joining.
            added = len(piece) + (1 if current else 0)

            if cur_len + added > self.chunk_size and current:
                chunks.append("\n".join(current))

                # Trim from the front until we're within the overlap budget.
                # NOTE (learning): we pop whole pieces, so actual overlap may
                # be slightly less than `self.overlap` — that is acceptable.
                while current and cur_len > self.overlap:
                    current.pop(0)
                    cur_len = (
                        sum(len(p) for p in current) + max(0, len(current) - 1)
                    )

            current.append(piece)
            cur_len = sum(len(p) for p in current) + max(0, len(current) - 1)

        if current:
            chunks.append("\n".join(current))

        return chunks
