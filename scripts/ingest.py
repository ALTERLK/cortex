"""Index a folder of documents (.md, .txt, .pdf) into the vector store.

Ingest is incremental: files whose content has not changed since the last
run are skipped, so re-running after editing one file embeds only that file.

Usage:
    uv run python scripts/ingest.py corpus
    uv run python scripts/ingest.py my_notes --chunk-size 300 --overlap 40
"""

import argparse
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.config import get_settings
from cortex.ingest.embedder import Embedder
from cortex.ingest.pipeline import ingest_directory
from cortex.ingest.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into the vector store.")
    parser.add_argument("directory", help="Folder to index (recursive)")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--store-path", default=get_settings().store_path)
    args = parser.parse_args()

    print("Loading the embedding model (the first run downloads BGE-M3, ~2.3 GB)…")
    embedder = Embedder()

    store = VectorStore(args.store_path)
    store.ensure_collection(embedder.dimension)

    print(f"Indexing '{args.directory}' into {args.store_path}")
    total = ingest_directory(
        Path(args.directory), store, embedder,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )
    print(f"\nDone. {total} chunks written; the index now holds {store.count()}.")


if __name__ == "__main__":
    main()
