"""Eval runner: answer every question with a pipeline, score it, aggregate.

Two kinds of numbers come out of a run:
  - quality   — did retrieval find the right document, is the answer correct;
  - cost      — LLM calls, tool calls, tokens, latency.
Reporting both is the point: a pipeline that scores higher but costs five
times as much is a trade-off, not an improvement.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from cortex.eval.dataset import QAPair
from cortex.eval.metrics import (
    compute_retrieval_hit,
    hit_rate,
    judge_answer,
    mrr,
)
from cortex.eval.pipelines import Pipeline, PipelineOutput
from cortex.llm.base import LLMClient, LLMUnavailableError
from cortex.llm.metering import MeteredLLM
from cortex.rag.citations import check_citations, is_refusal


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
    # Which pipeline answered ("rag" or "agent") and what it cost. The cost
    # fields are 0 in runs saved before they were recorded.
    mode: str = "rag"
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    # True when the answer cites at least one retrieved passage and no
    # passage number that does not exist. None for runs that predate the check.
    grounded: bool | None = None


@dataclass
class EvalReport:
    """Aggregated metrics across the full eval set.

    Retrieval metrics (hit-rate, MRR) aggregate ANSWERABLE questions only —
    an unanswerable trap question has no correct source to retrieve, so
    counting it would distort the retrieval picture in either direction.
    """

    results: list[EvalResult]
    meta: dict = field(default_factory=dict)

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
        return _mean([r.correctness for r in self.results])

    @property
    def avg_faithfulness(self) -> float:
        return _mean([r.faithfulness for r in self.results])

    @property
    def refusal_accuracy(self) -> float | None:
        """Share of unanswerable questions that were honestly refused."""
        traps = [r for r in self.results if not r.answerable]
        if not traps:
            return None
        return sum(1 for r in traps if r.correctness >= 4.0) / len(traps)

    @property
    def grounded_rate(self) -> float | None:
        """Share of answerable questions whose answer has valid citations."""
        checked = [r for r in self._answerable if r.grounded is not None]
        if not checked:
            return None
        return sum(1 for r in checked if r.grounded) / len(checked)

    @property
    def has_cost(self) -> bool:
        """False for runs saved before token usage was recorded."""
        return any(r.input_tokens for r in self.results)

    @property
    def agent_share(self) -> float:
        """Share of questions answered by the agent pipeline."""
        return _mean([1.0 if r.mode == "agent" else 0.0 for r in self.results])

    def avg_cost_usd(self) -> float | None:
        """Mean LLM cost per question, from the prices stored with the run."""
        if not self.has_cost:
            return None
        price_in = self.meta.get("price_input_per_m", 0.0) / 1_000_000
        price_out = self.meta.get("price_output_per_m", 0.0) / 1_000_000
        return _mean([r.input_tokens * price_in + r.output_tokens * price_out for r in self.results])

    def by_category(self) -> dict[str, dict[str, float]]:
        """Per-category breakdown: count, hit-rate, MRR, correctness, agent share."""
        out: dict[str, dict[str, float]] = {}
        for cat in sorted({r.category for r in self.results}):
            rows = [r for r in self.results if r.category == cat]
            answerable = [r for r in rows if r.answerable]
            out[cat] = {
                "count": len(rows),
                "hit_rate": hit_rate([r.hit for r in answerable]) if answerable else float("nan"),
                "mrr": mrr([r.reciprocal_rank for r in answerable]) if answerable else float("nan"),
                "correctness": _mean([r.correctness for r in rows]),
                "agent_share": _mean([1.0 if r.mode == "agent" else 0.0 for r in rows]),
                "avg_tokens": _mean([r.input_tokens + r.output_tokens for r in rows]),
            }
        return out

    def summary(self) -> dict:
        """Aggregate numbers as a JSON-safe dict (NaN becomes None)."""
        has_cost = self.has_cost
        return _json_safe({
            "questions": len(self.results),
            "answerable": len(self._answerable),
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "correctness": self.avg_correctness,
            "faithfulness": self.avg_faithfulness,
            "refusal_accuracy": self.refusal_accuracy,
            "grounded_rate": self.grounded_rate,
            "agent_share": self.agent_share,
            "avg_llm_calls": _mean([r.llm_calls for r in self.results]) if has_cost else None,
            "avg_tool_calls": _mean([r.tool_calls for r in self.results]) if has_cost else None,
            "avg_input_tokens": _mean([r.input_tokens for r in self.results]) if has_cost else None,
            "avg_output_tokens": _mean([r.output_tokens for r in self.results]) if has_cost else None,
            "avg_latency_ms": _mean([r.latency_ms for r in self.results]) if has_cost else None,
            "avg_cost_usd": self.avg_cost_usd(),
            "by_category": self.by_category(),
        })

    def print(self) -> None:
        """Print a human-readable summary."""
        n = len(self.results)
        n_ans = len(self._answerable)
        print(f"\n{'=' * 66}")
        print(f"  CORTEX EVAL REPORT  ({n} questions, {n_ans} answerable)")
        print(f"{'=' * 66}")
        print(f"  Hit-rate     : {self.hit_rate:.1%}  ({sum(r.hit for r in self._answerable)}/{n_ans})")
        print(f"  MRR          : {self.mrr:.3f}")
        print(f"  Correctness  : {self.avg_correctness:.2f} / 5")
        print(f"  Faithfulness : {self.avg_faithfulness:.2f} / 5")
        if self.refusal_accuracy is not None:
            print(f"  Refusals     : {self.refusal_accuracy:.0%} of unanswerable questions refused")
        if self.grounded_rate is not None:
            print(f"  Grounded     : {self.grounded_rate:.0%} of answers have valid citations")
        if self.has_cost:
            s = self.summary()
            print(f"  {'-' * 62}")
            print(f"  LLM calls    : {s['avg_llm_calls']:.2f} / question")
            print(f"  Tool calls   : {s['avg_tool_calls']:.2f} / question")
            tokens_in, tokens_out = s["avg_input_tokens"], s["avg_output_tokens"]
            print(f"  Tokens       : {tokens_in:.0f} in + {tokens_out:.0f} out / question")
            print(f"  Latency      : {s['avg_latency_ms'] / 1000:.1f} s / question")
            print(f"  Cost         : ${s['avg_cost_usd']:.5f} / question")
            print(f"  Agent share  : {self.agent_share:.0%} of questions")
        print(f"  {'-' * 62}")
        print(f"  {'category':<14}{'n':>4}{'hit-rate':>10}{'MRR':>8}{'correct':>9}{'agent':>8}")
        for cat, m in self.by_category().items():
            hr = f"{m['hit_rate']:.0%}" if not math.isnan(m["hit_rate"]) else "—"
            mr = f"{m['mrr']:.3f}" if not math.isnan(m["mrr"]) else "—"
            row = f"{cat:<14}{m['count']:>4}{hr:>10}{mr:>8}{m['correctness']:>8.2f}"
            print(f"  {row}{m['agent_share']:>8.0%}")
        print(f"{'=' * 66}\n")

    def save(self, path: Path | str) -> None:
        """Save the run — metadata, summary, and per-question rows — as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in self.results:
            row = asdict(r)
            row["answer_preview"] = row.pop("generated_answer")[:300]
            rows.append(row)
        payload = {"meta": self.meta, "summary": self.summary(), "results": rows}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> EvalReport:
        """Load a saved run.

        Accepts the current {"meta", "results"} layout and the original
        layout, a bare list of rows; fields a row lacks take their defaults.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data["results"] if isinstance(data, dict) else data
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        known = {f.name for f in fields(EvalResult)}
        results = []
        for row in rows:
            row = dict(row)
            row["generated_answer"] = row.pop("answer_preview", row.get("generated_answer", ""))
            results.append(EvalResult(**{k: v for k, v in row.items() if k in known}))
        return cls(results, meta)


def run_eval(
    dataset: list[QAPair],
    pipeline: Pipeline,
    judge_llm: LLMClient,
    *,
    top_k: int = 5,
    meter: MeteredLLM | None = None,
    meta: dict | None = None,
    verbose: bool = True,
) -> EvalReport:
    """Run *pipeline* over *dataset* and score every answer.

    Args:
        pipeline:  The system under test (RAG, agent, or auto-routed).
        judge_llm: Grades answer correctness. It is the same model as the
                   system under test here; a separate, stronger judge would
                   remove the self-grading bias.
        meter:     If the pipeline's LLM client is wrapped in a MeteredLLM,
                   pass it here to record tokens and LLM calls per question.
                   The judge must use an unmetered client so grading cost is
                   not charged to the pipeline.
    """
    results: list[EvalResult] = []
    n = len(dataset)

    for i, qa in enumerate(dataset, 1):
        if verbose:
            print(f"[{i:2d}/{n}] ({qa.category}) {qa.question[:55]}…")

        out, latency_ms = _answer_with_retry(pipeline, qa, meter)

        # RAG is scored on its top-k. The agent has no single ranked list —
        # it is scored on every passage it retrieved during the run, so its
        # hit-rate reads as "found the document at some point", not hit@k.
        depth = top_k if out.mode == "rag" else len(out.passages)
        hit, rr = compute_retrieval_hit(out.passages, qa.expected_sources, depth)

        # Unanswerable questions are scored deterministically: the only
        # correct behaviour is an honest refusal — no judge needed.
        if not qa.answerable:
            refused = is_refusal(out.answer)
            scores = {
                "correctness": 5.0 if refused else 1.0,
                "faithfulness": 5.0 if refused else 1.0,
            }
        else:
            scores = _retrying(lambda out=out, qa=qa: judge_answer(
                qa.question, qa.expected_answer, out.answer, judge_llm))

        results.append(
            EvalResult(
                question=qa.question,
                hit=hit,
                reciprocal_rank=rr,
                generated_answer=out.answer,
                correctness=scores["correctness"],
                faithfulness=scores["faithfulness"],
                category=qa.category,
                answerable=qa.answerable,
                retrieved_sources=[p.source for p in out.passages],
                mode=out.mode,
                llm_calls=meter.calls if meter else 0,
                tool_calls=out.tool_calls,
                input_tokens=meter.input_tokens if meter else 0,
                output_tokens=meter.output_tokens if meter else 0,
                latency_ms=latency_ms,
                grounded=check_citations(out.answer, len(out.passages)).grounded,
            )
        )

        if verbose:
            marker = "✓" if hit else "✗"
            cost = f"  {meter.input_tokens + meter.output_tokens} tok" if meter else ""
            print(
                f"       {marker} hit  rr={rr:.2f}  correct={scores['correctness']:.0f}/5  "
                f"faith={scores['faithfulness']:.0f}/5  [{out.mode}]{cost}  {latency_ms / 1000:.1f}s"
            )

    return EvalReport(results, meta or {})


# A run makes hundreds of LLM calls over many minutes; one dropped
# connection should cost a retry, not the whole run.
_RETRY_DELAYS_S = (5, 20, 60)


def _retrying[T](call: Callable[[], T]) -> T:
    """Run *call*, retrying when the LLM provider is unreachable."""
    for delay in _RETRY_DELAYS_S:
        try:
            return call()
        except LLMUnavailableError as exc:
            print(f"       provider unavailable ({str(exc)[:80]}); retrying in {delay}s")
            time.sleep(delay)
    return call()  # last attempt: let the error propagate


def _answer_with_retry(
    pipeline: Pipeline, qa: QAPair, meter: MeteredLLM | None
) -> tuple[PipelineOutput, float]:
    """Answer one question; cost and latency are those of the successful attempt."""

    def attempt() -> tuple[PipelineOutput, float]:
        if meter is not None:
            meter.reset()  # a failed attempt must not be charged to the answer
        started = time.perf_counter()
        out = pipeline.run(qa)
        return out, round((time.perf_counter() - started) * 1000, 1)

    return _retrying(attempt)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _json_safe(value: object) -> object:
    """Replace NaN with None, recursively — NaN is not valid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value
