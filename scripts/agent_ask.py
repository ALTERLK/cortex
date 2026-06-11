"""M4 agent CLI: ask a question using the agentic tool-use loop.

The LLM decides whether to search, how many times, and what to search for.
Watch the tool-call log to see its reasoning process.

Requires documents to be ingested first:
    uv run python scripts/ingest.py docs/

Usage:
    uv run python scripts/agent_ask.py "Compare chunking strategies"
    uv run python scripts/agent_ask.py "What evaluation metrics does Cortex use?" --max-iter 5
"""

import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.agent.loop import AgentLoop
from cortex.agent.tools import ToolExecutor
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.rag.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Q&A over your documents.")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--store-path", default="data/qdrant")
    args = parser.parse_args()

    print("Loading components…")
    embedder = Embedder()
    store = VectorStore(args.store_path)
    retriever = Retriever(store, embedder, top_k=args.top_k)
    executor = ToolExecutor(retriever, top_k=args.top_k)
    agent = AgentLoop(get_llm_client(), executor, max_iterations=args.max_iter)

    print(f"\nQuestion: {args.question}\n")
    result = agent.run(args.question)

    # --- tool call log ---
    if result.tool_calls:
        print(f"--- Agent made {len(result.tool_calls)} search(es) ---")
        for i, tc in enumerate(result.tool_calls, 1):
            q = tc.arguments.get("query", tc.arguments)
            print(f"  [{i}] {tc.name}({q!r})")
        print()

    # --- final answer ---
    print("=" * 60)
    from cortex.llm.postprocess import strip_thinking
    print(strip_thinking(result.answer))
    print("=" * 60)

    usage = result.usage
    print(f"\nIterations: {result.iterations} | Tokens: {usage.input_tokens} in / {usage.output_tokens} out")


if __name__ == "__main__":
    main()
