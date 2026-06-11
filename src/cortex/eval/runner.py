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
from cortex.rag.generator import Generator
from cortex.rag.retriever import Retriever


@dataclass
class EvalResult:
    """Metrics for one question."""

    question: str
    hit: bool
    reciprocal_rank: float
    generated_answer: str
    correctness: float
    faithfulness: float
    retrieved_sources: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated metrics across the full eval set."""

    results: list[EvalResult]

    @property
    def hit_rate(self) -> float:
        return hit_rate([r.hit for r in self.results])

    @property
    def mrr(self) -> float:
        return mrr([r.reciprocal_rank for r in self.results])

    @property
    def avg_correctness(self) -> float:
        vals = [r.correctness for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_faithfulness(self) -> float:
        vals = [r.faithfulness for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0

    def print(self) -> None:
        """Print a human-readable summary."""
        n = len(self.results)
        print(f"\n{'='*50}")
        print(f"  CORTEX EVAL REPORT  ({n} questions)")
        print(f"{'='*50}")
        print(f"  Hit-rate@k   : {self.hit_rate:.1%}  ({sum(r.hit for r in self.results)}/{n})")
        print(f"  MRR          : {self.mrr:.3f}")
        print(f"  Correctness  : {self.avg_correctness:.2f} / 5")
        print(f"  Faithfulness : {self.avg_faithfulness:.2f} / 5")
        print(f"{'='*50}\n")

    def save(self, path: Path | str = "data/eval_results.json") -> None:
        """Save detailed per-question results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "question": r.question,
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
            print(f"[{i:2d}/{n}] {qa.question[:60]}…")

        # Retrieve
        passages = retriever.retrieve(qa.question)
        hit, rr = compute_retrieval_hit(passages, qa.expected_sources, top_k)

        # Generate
        gen_response = generator.generate(qa.question, passages)

        # Judge
        scores = judge_answer(
            qa.question, qa.expected_answer, gen_response.answer, judge_llm
        )

        results.append(
            EvalResult(
                question=qa.question,
                hit=hit,
                reciprocal_rank=rr,
                generated_answer=gen_response.answer,
                correctness=scores["correctness"],
                faithfulness=scores["faithfulness"],
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
