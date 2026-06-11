"""M2 ingest CLI: walk a directory and load all documents into Qdrant.

Usage:
    uv run python scripts/ingest.py docs/
    uv run python scripts/ingest.py docs/ --chunk-size 300 --overlap 40
"""

import argparse
from pathlib import Path

from cortex.ingest.embedder import Embedder
from cortex.ingest.pipeline import ingest_directory
from cortex.ingest.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("directory", help="Folder to ingest (recursive)")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--store-path", default="data/qdrant")
    args = parser.parse_args()

    print("Loading embedding model (first run downloads ~570 MB)…")
    embedder = Embedder()

    store = VectorStore(args.store_path)
    store.ensure_collection(embedder.dimension)

    print(f"Ingesting '{args.directory}' → {args.store_path}")
    total = ingest_directory(
        Path(args.directory),
        store,
        embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"\nDone. {total} chunks stored.")


if __name__ == "__main__":
    main()
