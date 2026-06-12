"""Eval runner: iterate over the dataset, retrieve, generate, judge, aggregate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cortex.eval.dataset import QAPair
from cortex.eval.metrics import (
    compute_retrieval_hit,
    hit_rate,
    judge_answer,
    mrr,
)
from cortex.llm.base import LLMClient
from cortex.llm.postprocess import strip_thinking
from cortex.rag.generator import Generator
from cortex.rag.retriever import Retriever

# Phrases that signal an honest refusal. The generator's system prompt
# mandates the first one; the others catch close paraphrases.
_REFUSAL_MARKERS = (
    "don't have information",
    "do not have information",
    "no information about",
    "not covered in the provided documents",
)


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


@dataclass
class EvalResult:
    """Metrics for one question."""

    question: str
    hit: bool
    reciprocal_rank: float
    generated_answer: str
    correctness: float
    faithfulness: float
    category: str = "factual"
    answerable: bool = True
    retrieved_sources: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated metrics across the full eval set.

    Retrieval metrics (hit-rate, MRR) aggregate ANSWERABLE questions only —
    an unanswerable trap question has no correct source to retrieve, so
    counting it would distort the retrieval picture in either direction.
    """

    results: list[EvalResult]

    @property
    def _answerable(self) -> list[EvalResult]:
        return [r for r in self.results if r.answerable]

    @property
    def hit_rate(self) -> float:
        return hit_rate([r.hit for r in self._answerable])

    @property
    def mrr(self) -> float:
        return mrr([r.reciprocal_rank for r in self._answerable])

    @property
    def avg_correctness(self) -> float:
        vals = [r.correctness for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_faithfulness(self) -> float:
        vals = [r.faithfulness for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def refusal_accuracy(self) -> float | None:
        """Share of unanswerable questions that were honestly refused."""
        traps = [r for r in self.results if not r.answerable]
        if not traps:
            return None
        return sum(1 for r in traps if r.correctness >= 4.0) / len(traps)

    def by_category(self) -> dict[str, dict[str, float]]:
        """Per-category breakdown: count, hit-rate, MRR, correctness."""
        out: dict[str, dict[str, float]] = {}
        for cat in sorted({r.category for r in self.results}):
            rows = [r for r in self.results if r.category == cat]
            answerable = [r for r in rows if r.answerable]
            out[cat] = {
                "count": len(rows),
                "hit_rate": hit_rate([r.hit for r in answerable]) if answerable else float("nan"),
                "mrr": mrr([r.reciprocal_rank for r in answerable]) if answerable else float("nan"),
                "correctness": sum(r.correctness for r in rows) / len(rows),
            }
        return out

    def print(self) -> None:
        """Print a human-readable summary."""
        n = len(self.results)
        n_ans = len(self._answerable)
        print(f"\n{'=' * 62}")
        print(f"  CORTEX EVAL REPORT  ({n} questions, {n_ans} answerable)")
        print(f"{'=' * 62}")
        print(f"  Hit-rate@k   : {self.hit_rate:.1%}  ({sum(r.hit for r in self._answerable)}/{n_ans})")
        print(f"  MRR          : {self.mrr:.3f}")
        print(f"  Correctness  : {self.avg_correctness:.2f} / 5")
        print(f"  Faithfulness : {self.avg_faithfulness:.2f} / 5")
        if self.refusal_accuracy is not None:
            print(f"  Refusals     : {self.refusal_accuracy:.0%} of unanswerable questions refused")
        print(f"  {'-' * 58}")
        print(f"  {'category':<14}{'n':>4}{'hit-rate':>10}{'MRR':>8}{'correct':>9}")
        for cat, m in self.by_category().items():
            hr = f"{m['hit_rate']:.0%}" if m["hit_rate"] == m["hit_rate"] else "—"
            mr = f"{m['mrr']:.3f}" if m["mrr"] == m["mrr"] else "—"
            print(f"  {cat:<14}{m['count']:>4}{hr:>10}{mr:>8}{m['correctness']:>8.2f}")
        print(f"{'=' * 62}\n")

    def save(self, path: Path | str = "data/eval_results.json") -> None:
        """Save detailed per-question results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "question": r.question,
                "category": r.category,
                "answerable": r.answerable,
                "hit": r.hit,
                "reciprocal_rank": r.reciprocal_rank,
                "correctness": r.correctness,
                "faithfulness": r.faithfulness,
                "retrieved_sources": r.retrieved_sources,
                "answer_preview": r.generated_answer[:200],
            }
            for r in self.results
        ]
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved to {path}")


def run_eval(
    dataset: list[QAPair],
    retriever: Retriever,
    generator: Generator,
    judge_llm: LLMClient,
    *,
    top_k: int = 5,
    rewriter: object | None = None,
    verbose: bool = True,
) -> EvalReport:
    """Run the full evaluation loop and return an EvalReport.

    NOTE (learning): generator and judge use the same LLM here because we
    only have one configured provider. In a production eval, you would use
    a stronger, cheaper model as judge (e.g. Claude Sonnet) separate from
    the system-under-test to avoid self-grading bias.
    """
    results: list[EvalResult] = []
    n = len(dataset)

    for i, qa in enumerate(dataset, 1):
        if verbose:
            print(f"[{i:2d}/{n}] ({qa.category}) {qa.question[:55]}…")

        # Retrieve. With a rewriter, history-dependent questions are first
        # rewritten into standalone queries (the A/B for query rewriting).
        query = qa.question
        if rewriter is not None and qa.history:
            query = rewriter.rewrite(qa.question, qa.history)
            if verbose and query != qa.question:
                print(f"       rewritten: {query[:60]}")
        passages = retriever.retrieve(query)
        hit, rr = compute_retrieval_hit(passages, qa.expected_sources, top_k)

        # Generate
        gen_response = generator.generate(qa.question, passages, qa.history or None)
        answer = strip_thinking(gen_response.answer)

        # Score. Unanswerable questions are scored deterministically: the
        # only correct behaviour is an honest refusal — no judge needed.
        if not qa.answerable:
            refused = _is_refusal(answer)
            scores = {
                "correctness": 5.0 if refused else 1.0,
                "faithfulness": 5.0 if refused else 1.0,
            }
        else:
            scores = judge_answer(qa.question, qa.expected_answer, answer, judge_llm)

        results.append(
            EvalResult(
                question=qa.question,
                hit=hit,
                reciprocal_rank=rr,
                generated_answer=answer,
                correctness=scores["correctness"],
                faithfulness=scores["faithfulness"],
                category=qa.category,
                answerable=qa.answerable,
                retrieved_sources=[p.source for p in passages],
            )
        )

        if verbose:
            marker = "✓" if hit else "✗"
            print(
                f"       {marker} hit  rr={rr:.2f}  "
                f"correct={scores['correctness']:.0f}/5  faith={scores['faithfulness']:.0f}/5"
            )

    return EvalReport(results)
