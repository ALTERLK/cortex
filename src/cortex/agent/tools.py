"""Tool definitions and execution for the agent loop.

A "tool" here has two parts:
  1. A JSON schema that tells the LLM what the tool does and what arguments it takes.
  2. An execute() function that actually runs the tool and returns a string result.

NOTE (learning): the schema is what the LLM reads; the execute function is what
WE run when the LLM asks for it. The LLM never runs code — it just asks us to.
"""

from __future__ import annotations

from cortex.ingest.store import SearchResult
from cortex.rag.retriever import Retriever

# ---------------------------------------------------------------------------
# Tool schemas (sent to the LLM)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the personal knowledge base for passages relevant to a query. "
                "Call this whenever you need information to answer the question. "
                "You may call it multiple times with different queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific, focused search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all source documents available in the knowledge base.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Runs tool calls requested by the LLM and returns string results.

    Args:
        retriever:    Retriever instance connected to the Qdrant store.
        top_k:        Number of passages to return per search.
    """

    def __init__(self, retriever: Retriever, top_k: int = 4) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def execute(self, name: str, arguments: dict) -> str:
        if name == "search_knowledge_base":
            return self._search(arguments.get("query", ""))
        if name == "list_documents":
            return self._list_documents()
        return f"[Unknown tool: {name}]"

    def _search(self, query: str) -> str:
        if not query:
            return "[Error: empty query]"
        results = self._retriever.retrieve(query)
        if not results:
            return "[No relevant passages found]"
        return _format_search_results(results)

    def _list_documents(self) -> str:
        sources = self._retriever._store.list_sources()
        if not sources:
            return "[Knowledge base is empty]"
        return "Available documents:\n" + "\n".join(f"- {s}" for s in sources)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_search_results(results: list[SearchResult]) -> str:
    """Format search results as a numbered list for the LLM to read."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] (source: {r.source}, score: {r.score:.3f})\n{r.text}")
    return "\n\n".join(lines)
