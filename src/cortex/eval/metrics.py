"""Evaluation metrics: hit-rate@k, MRR, and LLM-as-judge.

NOTE (learning): these are the two standard axes for RAG evaluation:
  - Retrieval quality: did the right chunks come back?  (hit-rate, MRR)
  - Answer quality: is the generated text correct and grounded? (LLM-as-judge)

Running both is what separates a serious RAG project from a demo.
"""

from __future__ import annotations

import json
import re

from cortex.ingest.store import SearchResult
from cortex.llm.base import LLMClient


# ---------------------------------------------------------------------------
# Retrieval metrics (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------


def hit_rate(hits: list[bool]) -> float:
    """Fraction of questions where a relevant chunk appeared in top-k."""
    return sum(hits) / len(hits) if hits else 0.0


def mrr(reciprocal_ranks: list[float]) -> float:
    """Mean Reciprocal Rank across all questions."""
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_retrieval_hit(
    retrieved: list[SearchResult],
    expected_sources: list[str],
    top_k: int,
) -> tuple[bool, float]:
    """Return (hit, reciprocal_rank) for one question.

    A hit is True if any chunk in the top-k comes from an expected source.
    Reciprocal rank is 1/rank of the first hit, or 0 if no hit.
    """
    for rank, result in enumerate(retrieved[:top_k], 1):
        if result.source in expected_sources:
            return True, 1.0 / rank
    return False, 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator of question-answering systems.

Given:
  - A question
  - A reference answer (ground truth)
  - A generated answer from the system under test

Score the generated answer on two dimensions (integer 1–5):

  correctness  — Does it correctly and completely answer the question?
                 1 = completely wrong, 5 = fully correct and complete
  faithfulness — Does it only state things supported by the sources it cites?
                 1 = fabricates information, 5 = every claim is grounded

Question:
{question}

Reference answer:
{expected}

Generated answer:
{generated}

Reply with ONLY a JSON object, no extra text:
{{"correctness": <1-5>, "faithfulness": <1-5>}}"""


def judge_answer(
    question: str,
    expected_answer: str,
    generated_answer: str,
    llm: LLMClient,
) -> dict[str, float]:
    """Ask the LLM to score a generated answer.  Returns correctness + faithfulness.

    NOTE (learning): using an LLM to grade LLM output ("LLM-as-judge") is the
    current industry standard for answer quality at scale. It is not perfect —
    the judge can share biases with the generator — but it is far more
    practical than human annotation for a 20-50 item eval set.
    """
    # Strip <thinking> blocks that extended-thinking models prepend.
    clean = re.sub(r"<thinking>.*?</thinking>\s*", "", generated_answer, flags=re.DOTALL).strip()

    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected=expected_answer,
        generated=clean or "(no answer)",
    )
    response = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return _parse_judge_response(response.content or "")


def _parse_judge_response(text: str) -> dict[str, float]:
    """Extract {correctness, faithfulness} from the judge's response.

    Handles:
    - Plain JSON
    - JSON wrapped in a markdown code fence
    - <thinking> blocks before the JSON
    """
    # Strip thinking blocks
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
    # Extract first JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "correctness": float(data.get("correctness", 2.5)),
                "faithfulness": float(data.get("faithfulness", 2.5)),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: couldn't parse
    return {"correctness": 2.5, "faithfulness": 2.5}
