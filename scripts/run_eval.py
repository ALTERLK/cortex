"""Run the evaluation harness and print a scored report.

Usage:
    uv run python scripts/run_eval.py --mode rag                    # single-shot RAG
    uv run python scripts/run_eval.py --mode agent                  # tool-use loop
    uv run python scripts/run_eval.py --mode auto                   # router picks per question
    uv run python scripts/run_eval.py --retriever hybrid_rerank --rewrite
    uv run python scripts/run_eval.py --mode agent --limit 5        # quick smoke run

Every run is saved under benchmarks/results/ and shows up on the /eval dashboard.
"""

import argparse
import io
import sys
from datetime import date

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.agent.loop import AgentLoop
from cortex.agent.router import QueryRouter
from cortex.agent.tools import ToolExecutor
from cortex.config import get_settings
from cortex.eval.dataset import load_dataset
from cortex.eval.pipelines import AgentPipeline, AutoPipeline, RagPipeline
from cortex.eval.runner import run_eval
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.llm.metering import MeteredLLM
from cortex.rag.factory import STRATEGIES, build_retriever
from cortex.rag.generator import Generator
from cortex.rag.rewriter import QueryRewriter


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Cortex evaluation harness.")
    parser.add_argument("--dataset", default="benchmarks/eval_set.json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["rag", "agent", "auto"], default="rag",
                        help="Answer pipeline to evaluate")
    parser.add_argument("--retriever", choices=STRATEGIES, default="hybrid",
                        help="Retrieval strategy (A/B on the same dataset)")
    parser.add_argument("--rewrite", action="store_true",
                        help="Rewrite history-dependent questions into standalone queries (rag path)")
    parser.add_argument("--auto-strategy", choices=["escalate", "classify"],
                        default=settings.auto_strategy,
                        help="How --mode auto chooses a pipeline (see docs/evaluation.md)")
    parser.add_argument("--store-path", default=settings.store_path)
    parser.add_argument("--name", default=None, help="Result file name (default: derived from the options)")
    parser.add_argument("--label", default=None, help="Display label on the dashboard")
    parser.add_argument("--group", default="mode", help="Dashboard group: 'retrieval' or 'mode'")
    parser.add_argument("--order", type=int, default=0, help="Position within the dashboard group")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Eval set: {len(dataset)} questions  mode={args.mode}  retriever={args.retriever}"
          f"  rewrite={args.rewrite}  top_k={args.top_k}")
    print("Loading components…")

    embedder = Embedder()
    store = VectorStore(args.store_path)
    retriever = build_retriever(store, embedder, args.retriever, top_k=args.top_k)

    # The system under test talks to the LLM through a meter, so every call
    # it makes — routing, rewriting, each agent step — is counted. The judge
    # uses the bare client: grading is not part of the system's cost.
    judge = get_llm_client()
    meter = MeteredLLM(judge)

    rag = RagPipeline(retriever, Generator(meter), QueryRewriter(meter) if args.rewrite else None)
    agent = AgentPipeline(AgentLoop(
        meter, ToolExecutor(retriever, store),
        max_iterations=settings.agent_max_iterations,
        token_budget=settings.agent_token_budget,
        compact_at=settings.agent_compact_at,
    ))
    router = QueryRouter(meter) if args.auto_strategy == "classify" else None
    pipeline = {"rag": rag, "agent": agent, "auto": AutoPipeline(rag, agent, router)}[args.mode]

    mode_name = f"auto-{args.auto_strategy}" if args.mode == "auto" else args.mode
    name = args.name or f"{mode_name}-{args.retriever}" + ("-rewrite" if args.rewrite else "")
    meta = {
        "label": args.label or name,
        "group": args.group,
        "order": args.order,
        "date": date.today().isoformat(),
        "mode": args.mode,
        "auto_strategy": args.auto_strategy if args.mode == "auto" else None,
        "retriever": args.retriever,
        "rewrite": args.rewrite,
        "top_k": args.top_k,
        "model": settings.llm_model,
        "price_input_per_m": settings.llm_price_input_per_m,
        "price_output_per_m": settings.llm_price_output_per_m,
    }

    print()
    report = run_eval(dataset, pipeline, judge_llm=judge, top_k=args.top_k, meter=meter, meta=meta)
    report.print()
    suffix = f"-limit{args.limit}" if args.limit else ""
    report.save(f"benchmarks/results/{name}{suffix}.json" if not args.limit
                else f"data/eval-smoke/{name}{suffix}.json")


if __name__ == "__main__":
    main()
