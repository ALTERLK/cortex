"""Ask one question from the command line with single-shot RAG.

Shows the pipeline without the web server: retrieve, generate, print the
answer with its sources and the retrieval trace of each passage.

Usage:
    uv run python examples/ask.py "What does a wildcard CORS origin prevent?"
    uv run python examples/ask.py "How do I cache dependencies in uv?" --retriever hybrid_rerank
"""

import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.config import get_settings
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.llm.postprocess import strip_thinking
from cortex.rag.factory import STRATEGIES, build_retriever
from cortex.rag.generator import Generator


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask a question over your documents.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=STRATEGIES, default=settings.retrieval_mode)
    args = parser.parse_args()

    print("Loading components…")
    embedder = Embedder()
    store = VectorStore(settings.store_path)
    retriever = build_retriever(store, embedder, args.retriever, top_k=args.top_k)

    passages = retriever.retrieve(args.question)
    if not passages:
        print("No passages found. Index some documents first: scripts/ingest.py <folder>")
        raise SystemExit(1)

    result = Generator(get_llm_client()).generate(args.question, passages)

    print("\n" + "=" * 70)
    print(strip_thinking(result.answer))
    print("=" * 70 + "\nSources:")
    for i, p in enumerate(passages, 1):
        trace = p.trace
        how = ", ".join(
            f"{label} {value}"
            for label, value in (
                ("dense #", trace.dense_rank), ("bm25 #", trace.sparse_rank),
                ("rrf", trace.rrf_score), ("rerank", trace.rerank_score),
            )
            if value is not None
        ) if trace else ""
        print(f"  [{i}] {p.source} (chunk {p.chunk_index})  {how}")
    print(f"\nTokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out")


if __name__ == "__main__":
    main()
