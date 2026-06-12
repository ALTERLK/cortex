"""M5 evaluation CLI: run the full eval harness and print a scored report.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --limit 5          # quick smoke run
    uv run python scripts/run_eval.py --top-k 3
"""

import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.eval.dataset import load_dataset
from cortex.eval.runner import run_eval
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.rag.generator import Generator
from cortex.rag.hybrid import HybridRetriever
from cortex.rag.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cortex evaluation harness.")
    parser.add_argument("--dataset", default="data/eval_set.json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=["dense", "hybrid", "hybrid_rerank"], default="dense",
                        help="Retrieval strategy to evaluate (A/B on the same dataset)")
    parser.add_argument("--rewrite", action="store_true",
                        help="Rewrite history-dependent questions into standalone queries")
    parser.add_argument("--store-path", default="data/qdrant")
    parser.add_argument("--save", default="data/eval_results.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Eval set: {len(dataset)} questions  top_k={args.top_k}  retriever={args.retriever}")
    print("Loading components…")

    embedder = Embedder()
    store = VectorStore(args.store_path)
    if args.retriever == "dense":
        retriever = Retriever(store, embedder, top_k=args.top_k)
    else:
        retriever = HybridRetriever(store, embedder, top_k=args.top_k)
        if args.retriever == "hybrid_rerank":
            from cortex.rag.reranker import Reranker, RerankingRetriever
            print("Loading reranker (first run downloads ~1.1 GB)…")
            retriever = RerankingRetriever(retriever, Reranker(), top_k=args.top_k)
    llm = get_llm_client()
    generator = Generator(llm)

    rewriter = None
    if args.rewrite:
        from cortex.rag.rewriter import QueryRewriter
        rewriter = QueryRewriter(llm)

    print()
    report = run_eval(dataset, retriever, generator, judge_llm=llm,
                      top_k=args.top_k, rewriter=rewriter)
    report.print()
    report.save(args.save)


if __name__ == "__main__":
    main()
