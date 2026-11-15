"""Agent tools: what the model may call, and the code that runs when it does.

A tool has two halves that must never drift apart:
  1. a JSON schema the model reads to decide when and how to call it, and
  2. a Python function we run when the model asks for it.

The `@tool` decorator keeps both in one place: the schema is declared right
above the function it describes, and `ToolExecutor` discovers every decorated
method at construction time. Adding a tool is adding one method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cortex.ingest.store import SearchResult, VectorStore

# JSON-schema type name -> Python type used to validate model-supplied arguments.
_JSON_TYPES: dict[str, type] = {"string": str, "integer": int}


@dataclass(frozen=True)
class ToolSpec:
    """The model-facing description of one tool."""

    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...]

    def schema(self) -> dict[str, Any]:
        """Render the spec in the OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                },
            },
        }

    def validate(self, arguments: dict[str, Any]) -> str | None:
        """Return a description of what is wrong with *arguments*, or None."""
        for name in self.required:
            if name not in arguments:
                return f"missing required argument '{name}'"
        for name, value in arguments.items():
            if name not in self.properties:
                expected = ", ".join(self.properties) or "none"
                return f"unexpected argument '{name}' (expected: {expected})"
            json_type = self.properties[name].get("type", "string")
            py_type = _JSON_TYPES.get(json_type)
            # bool is a subclass of int in Python; a JSON `true` is not an integer.
            if py_type and (not isinstance(value, py_type) or isinstance(value, bool)):
                return f"argument '{name}' must be of type {json_type}"
        return None


def tool(
    description: str,
    properties: dict[str, dict[str, Any]] | None = None,
    required: tuple[str, ...] = (),
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Mark a ToolExecutor method as a tool the model can call."""

    def decorate(fn: Callable[..., str]) -> Callable[..., str]:
        fn.tool_spec = ToolSpec(fn.__name__, description, properties or {}, required)  # type: ignore[attr-defined]
        return fn

    return decorate


class ToolError(Exception):
    """Raised inside a tool to report a problem back to the model."""


@dataclass
class RunContext:
    """State that belongs to ONE agent run.

    The ToolExecutor is created once and shared by every request, so anything
    that must not leak between concurrent runs lives here instead. Its job is
    the citation table: every passage the run has seen gets one number, and
    the same passage always gets the same number no matter which search found
    it. Without this, each search would restart at [1] and a citation in the
    final answer could point at two different passages.
    """

    sources: list[SearchResult] = field(default_factory=list)
    _numbers: dict[tuple[str, int], int] = field(default_factory=dict)

    def number_of(self, passage: SearchResult) -> int | None:
        """The citation number already assigned to *passage*, or None if unseen."""
        return self._numbers.get((passage.source, passage.chunk_index))

    def cite(self, passage: SearchResult) -> int:
        """Return the run-wide citation number for *passage*, assigning one if new."""
        key = (passage.source, passage.chunk_index)
        if key not in self._numbers:
            self.sources.append(passage)
            self._numbers[key] = len(self.sources)
        return self._numbers[key]


class ToolExecutor:
    """Runs the tool calls requested by the model and returns text results.

    Args:
        retriever: Any retriever (dense, hybrid, reranking) with retrieve().
        store:     The vector store, needed by the document-navigation tools.
        top_k:     Passages returned per search.
    """

    def __init__(
        self,
        retriever: object,
        store: VectorStore | None = None,
        top_k: int = 4,
    ) -> None:
        self._retriever = retriever
        self._store = store
        self._top_k = top_k
        self._tools: dict[str, tuple[ToolSpec, Callable[..., str]]] = {}
        # vars() preserves definition order, so tools reach the model in the
        # order they are written below.
        for attr, member in vars(type(self)).items():
            spec = getattr(member, "tool_spec", None)
            if isinstance(spec, ToolSpec):
                self._tools[spec.name] = (spec, getattr(self, attr))

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas to send to the model with every request."""
        return [spec.schema() for spec, _ in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any], ctx: RunContext | None = None) -> str:
        """Run one tool call and return its result as text.

        Failures are returned, not raised: an unknown tool, bad arguments, or
        a crash inside the tool all come back as an "Error: ..." observation.
        The model reads it on the next turn and can correct itself, which is
        better than aborting a run over one malformed call.
        """
        ctx = ctx if ctx is not None else RunContext()
        entry = self._tools.get(name)
        if entry is None:
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(self._tools)}."
        spec, handler = entry
        problem = spec.validate(arguments)
        if problem:
            return f"Error: {problem}."
        try:
            return handler(ctx, **arguments)
        except ToolError as exc:
            return f"Error: {exc}."
        except Exception as exc:  # noqa: BLE001 — a tool bug must not kill the run
            return f"Error: tool '{name}' failed ({exc})."

    # ------------------------------------------------------------------ tools

    @tool(
        "Search the knowledge base for passages relevant to a query. Use a specific, "
        "focused query. Call it again with a different query when the results do not "
        "cover the question; split multi-part questions into separate searches.",
        properties={"query": {"type": "string", "description": "A specific, focused search query."}},
        required=("query",),
    )
    def search_knowledge_base(self, ctx: RunContext, query: str) -> str:
        if not query.strip():
            raise ToolError("the query is empty")
        results = self._retriever.retrieve(query, top_k=self._top_k)
        if not results:
            return "No relevant passages found. Try a different query."

        # Passages the run already has are not printed again: repeating them
        # costs tokens on every later turn and teaches the model nothing.
        # Saying so explicitly is also the signal that searching has stopped
        # paying off — without it the model keeps rephrasing the same query.
        seen = [f"[{n}]" for p in results if (n := ctx.number_of(p)) is not None]
        new = [p for p in results if ctx.number_of(p) is None]
        if not new:
            return (
                f"No new passages: this search only returned passages you already have "
                f"({', '.join(seen)}). Answer from what you have, or search for a different aspect."
            )
        text = format_passages(new, ctx)
        if seen:
            text += f"\nAlso matched, already retrieved: {', '.join(seen)}."
        return text

    @tool(
        "Read the passages immediately before and after a passage you already retrieved. "
        "Use it when a passage is cut off or the answer seems to continue outside it — "
        "it is cheaper and more precise than searching again.",
        properties={
            "source": {"type": "string", "description": "The source path shown with the passage."},
            "chunk_index": {"type": "integer", "description": "The chunk number shown with the passage."},
            "window": {"type": "integer", "description": "Passages to read on each side (1-3, default 1)."},
        },
        required=("source", "chunk_index"),
    )
    def read_neighbors(self, ctx: RunContext, source: str, chunk_index: int, window: int = 1) -> str:
        if self._store is None:
            raise ToolError("document navigation is not available")
        window = max(1, min(window, 3))
        chunks = self._store.get_chunks(source, chunk_index - window, chunk_index + window)
        if not chunks:
            raise ToolError(f"no document '{source}' with chunk {chunk_index}; check the source path")
        return format_passages(chunks, ctx)

    @tool("List the documents available in the knowledge base.")
    def list_documents(self, ctx: RunContext) -> str:
        if self._store is None:
            raise ToolError("the document list is not available")
        sources = self._store.list_sources()
        if not sources:
            return "The knowledge base is empty."
        return "Available documents:\n" + "\n".join(f"- {s}" for s in sources)


def format_passages(passages: list[SearchResult], ctx: RunContext) -> str:
    """Render passages for the model, numbered with run-wide citation numbers.

    The passages are wrapped in <search_results> tags. Retrieved text is
    untrusted — an uploaded document can contain "ignore your instructions"
    — and the system prompt tells the model that everything inside these
    tags is quoted data, never instructions.
    """
    blocks = [
        f"[{ctx.cite(p)}] source: {p.source} · chunk {p.chunk_index}\n{p.text}"
        for p in passages
    ]
    return "<search_results>\n" + "\n\n".join(blocks) + "\n</search_results>"
