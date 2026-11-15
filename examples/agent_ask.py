"""Ask one question from the command line with the agent loop.

The model decides what to search for and how many times. Each step is
printed as it happens, followed by the answer and the run's cost.

Usage:
    uv run python examples/agent_ask.py "How do FastAPI and uv each handle caching?"
    uv run python examples/agent_ask.py "Compare path and query parameters" --max-iter 4
"""

import argparse
import sys

# UTF-8 output even on consoles whose default code page cannot encode the text.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cortex.agent.loop import AgentLoop, AgentResponse, ToolCallRecord
from cortex.agent.tools import ToolExecutor
from cortex.agent.trace import Span
from cortex.config import get_settings
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.llm.postprocess import strip_thinking
from cortex.rag.factory import build_retriever


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Agentic Q&A over your documents.")
    parser.add_argument("question")
    parser.add_argument("--max-iter", type=int, default=settings.agent_max_iterations)
    args = parser.parse_args()

    print("Loading components…")
    embedder = Embedder()
    store = VectorStore(settings.store_path)
    retriever = build_retriever(store, embedder, settings.retrieval_mode)
    agent = AgentLoop(get_llm_client(), ToolExecutor(retriever, store), max_iterations=args.max_iter)

    print(f"\nQuestion: {args.question}\n")
    result: AgentResponse | None = None
    for event in agent.run_events(args.question):
        if isinstance(event, Span):
            print(f"  llm   {event.name:<8} {event.ms / 1000:5.1f}s  "
                  f"{event.input_tokens} in / {event.output_tokens} out")
        elif isinstance(event, ToolCallRecord):
            print(f"  tool  {event.name}({event.arguments})  "
                  f"-> {len(event.new_sources)} new passage(s)")
        else:
            result = event

    assert result is not None
    print("\n" + "=" * 70)
    print(strip_thinking(result.answer))
    print("=" * 70 + "\nSources:")
    for i, p in enumerate(result.sources, 1):
        print(f"  [{i}] {p.source} (chunk {p.chunk_index})")
    print(f"\n{result.iterations} LLM calls, {len(result.tool_calls)} tool calls, "
          f"{result.usage.input_tokens} in / {result.usage.output_tokens} out tokens, "
          f"stopped: {result.stop_reason}")


if __name__ == "__main__":
    main()
