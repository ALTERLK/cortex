"""M3 ask CLI: ask a question, get a cited answer from your documents.

Requires documents to be ingested first:
    uv run python scripts/ingest.py docs/

Usage:
    uv run python scripts/ask.py "What is RAG?"
    uv run python scripts/ask.py "How does chunking affect retrieval quality?" --top-k 3
"""

import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.rag.generator import Generator
from cortex.rag.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over your documents.")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--store-path", default="data/qdrant")
    args = parser.parse_args()

    # --- set up components ---
    print("Loading components…")
    embedder = Embedder()
    store = VectorStore(args.store_path)
    retriever = Retriever(store, embedder, top_k=args.top_k)
    generator = Generator(get_llm_client())

    # --- retrieve ---
    print(f"\nSearching for: {args.question!r}")
    passages = retriever.retrieve(args.question)
    if not passages:
        print("No passages found. Have you run scripts/ingest.py yet?")
        raise SystemExit(1)

    # --- generate ---
    print("Generating answer…\n")
    result = generator.generate(args.question, passages)

    # --- display answer ---
    print("=" * 60)
    print(result.answer)
    print("=" * 60)

    # --- display sources ---
    print("\nSources:")
    for i, p in enumerate(passages, 1):
        print(f"  [{i}] {p.source}  (chunk {p.chunk_index}, score {p.score:.3f})")

    print(f"\nTokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out")


if __name__ == "__main__":
    main()
