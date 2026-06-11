"""M2 search demo: embed a query and print the top-k matching chunks.

Requires the Qdrant store to be populated first:
    uv run python scripts/ingest.py docs/

Usage:
    uv run python scripts/search_demo.py "what is retrieval augmented generation"
    uv run python scripts/search_demo.py "how does chunking work" --top-k 3
"""

import argparse
import io
import sys

# Force UTF-8 output on Windows (default console uses GBK which can't
# encode certain Unicode characters present in documents).
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the Qdrant knowledge base.")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--store-path", default="data/qdrant")
    args = parser.parse_args()

    embedder = Embedder()
    store = VectorStore(args.store_path)

    query_vector = embedder.embed([args.query])[0]
    results = store.search(query_vector, top_k=args.top_k)

    print(f"Query : {args.query!r}")
    print(f"Top-{args.top_k} results:\n")
    for i, r in enumerate(results):
        preview = r.text[:150].replace("\n", " ")
        ellipsis = "…" if len(r.text) > 150 else ""
        print(f"[{i+1}] score={r.score:.4f}  source={r.source}  chunk={r.chunk_index}")
        print(f"     {preview}{ellipsis}\n")


if __name__ == "__main__":
    main()
