"""M1 demo: load a document and print its chunks.

Usage:
    uv run python scripts/chunk_demo.py                  # uses docs/sample.md
    uv run python scripts/chunk_demo.py path/to/file.pdf
    uv run python scripts/chunk_demo.py --chunk-size 200 --overlap 30 docs/sample.md
"""

import argparse
from pathlib import Path

from cortex.ingest.chunker import RecursiveCharacterChunker
from cortex.ingest.loader import load_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a document and print results.")
    parser.add_argument("file", nargs="?", default="docs/sample.md")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    doc = load_document(Path(args.file))
    chunker = RecursiveCharacterChunker(
        chunk_size=args.chunk_size, overlap=args.overlap
    )
    chunks = chunker.chunk(doc)

    print(f"Document : {doc.source}")
    print(f"Length   : {len(doc.text)} chars")
    print(f"Settings : chunk_size={args.chunk_size}  overlap={args.overlap}")
    print(f"Chunks   : {len(chunks)}")
    print()

    for c in chunks:
        idx = c.metadata["chunk_index"]
        preview = c.text[:120].replace("\n", " ")
        ellipsis = "…" if len(c.text) > 120 else ""
        print(f"[{idx:02d}] {len(c.text):>4} chars │ {preview}{ellipsis}")

    print("\nDone.")


if __name__ == "__main__":
    main()
