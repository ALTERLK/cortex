"""Expose Cortex as an MCP server.

MCP (Model Context Protocol) is the open standard that lets AI clients —
coding assistants, desktop apps, IDEs — call external tools. Exposing the
knowledge base as MCP tools lets any such client search the documents or
get cited answers without going through the web API.

This file and the HTTP API share every component underneath (retriever,
generator) and differ only in transport — retrieval logic lives outside
both, so a second protocol is about a hundred lines of glue.

Run (stdio transport, started by the MCP client itself):
    uv run python scripts/mcp_serve.py
"""

from __future__ import annotations

import os
import sys
import time

# CRITICAL: in stdio transport, stdout IS the JSON-RPC channel. Any library
# that prints to stdout (HF progress bars, tqdm) corrupts the protocol and
# kills the connection. Silence them before the heavy imports happen.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from mcp.server.fastmcp import FastMCP

from cortex.config import get_settings
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.llm.postprocess import strip_thinking
from cortex.rag.factory import build_retriever
from cortex.rag.generator import Generator

mcp = FastMCP("cortex")

# Lazy singletons: the embedding model is only loaded on the first tool
# call, so the client's initial handshake (listing tools) stays instant.
_retriever: object | None = None
_generator: Generator | None = None


def _get_retriever() -> object:
    global _retriever
    if _retriever is None:
        settings = get_settings()
        t0 = time.time()
        # Diagnostics go to stderr: stdout carries the protocol.
        print("[cortex-mcp] loading embedder…", file=sys.stderr, flush=True)
        embedder = Embedder()
        store = VectorStore(path=settings.store_path)
        store.ensure_collection(embedder.dimension)
        # Same retrieval strategy as the web API, from the same config.
        _retriever = build_retriever(store, embedder, settings.retrieval_mode)
        print(f"[cortex-mcp] retriever ready in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    return _retriever


def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(get_llm_client())
    return _generator


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the Cortex knowledge base for passages relevant to a query.

    Returns the top matching passages with their source file and
    similarity score. Use this when you need raw source material.
    """
    results = _get_retriever().retrieve(query, top_k=top_k)
    if not results:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{i}] (source: {r.source}, score: {r.score:.3f})\n{r.text}"
        for i, r in enumerate(results, 1)
    )


@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    """Ask the Cortex knowledge base a question and get a cited answer.

    Runs the full RAG pipeline (retrieve + generate). Citations [N] refer
    to the source list appended after the answer. Requires LLM_API_KEY.
    """
    retriever = _get_retriever()
    passages = retriever.retrieve(question)
    result = _get_generator().generate(question, passages)

    sources = "\n".join(
        f"[{i}] {p.source} (chunk {p.chunk_index})"
        for i, p in enumerate(passages, 1)
    )
    answer = strip_thinking(result.answer)
    return f"{answer}\n\nSources:\n{sources}" if sources else answer


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
