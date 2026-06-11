"""Generator: turn retrieved passages + a question into a cited answer.

NOTE (learning): the Generator's job is prompt engineering — it decides how
to present the retrieved passages to the LLM and how to instruct it to cite
them. All LLM communication goes through the LLMClient protocol, so the
generator works with any provider (Claude, DeepSeek, GPT…).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cortex.ingest.store import SearchResult
from cortex.llm.base import LLMClient, TokenUsage


_SYSTEM_PROMPT = """\
You are Cortex, a personal knowledge assistant. You answer questions using ONLY
the context passages provided below. For every claim you make, cite the passage
number in square brackets — for example [1] or [2][3]. If the answer is not
present in the passages, say exactly: "I don't have information about that in
the provided documents." Do not fabricate or infer beyond what the passages say.\
"""


@dataclass
class GeneratorResponse:
    """The result of a single RAG query.

    Attributes:
        answer:   The LLM's response text, with inline [N] citations.
        passages: The retrieved passages that were given as context.
        usage:    Token counts for cost tracking.
    """

    answer: str
    passages: list[SearchResult] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))


class Generator:
    """Generates a cited answer from a question and retrieved passages.

    Args:
        llm: Any object satisfying the LLMClient protocol.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(self, question: str, passages: list[SearchResult]) -> GeneratorResponse:
        """Call the LLM with the question and passages; return a cited answer.

        If no passages are provided the LLM is still called — it will invoke
        the fallback phrase from the system prompt rather than hallucinating.
        """
        user_content = _build_user_message(question, passages)
        response = self._llm.chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,  # low temp for factual, grounded answers
        )
        return GeneratorResponse(
            answer=response.content or "",
            passages=passages,
            usage=response.usage,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_message(question: str, passages: list[SearchResult]) -> str:
    """Format passages as a numbered context block above the question.

    NOTE (learning): numbering from [1] lets the model use short inline
    citations without repeating source filenames mid-sentence. The source
    filename is shown in the legend printed after the answer so the user can
    open the original document.
    """
    if not passages:
        return f"Question: {question}"

    context_lines: list[str] = ["Context passages:"]
    for i, p in enumerate(passages, 1):
        context_lines.append(f"\n[{i}] (source: {p.source})\n{p.text}")

    context_block = "\n".join(context_lines)
    return f"{context_block}\n\nQuestion: {question}"
