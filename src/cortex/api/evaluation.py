"""Evaluation dashboard endpoint.

  GET /eval/results — every saved evaluation run, summarised

The numbers on the dashboard are computed here, by the same `EvalReport`
class the evaluation CLI uses, from the same result files it wrote. The
dashboard therefore cannot disagree with the CLI report: there is one
implementation of every metric.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from cortex.config import get_settings
from cortex.eval.runner import EvalReport

router = APIRouter()


@router.get("/eval/results")
def eval_results() -> dict:
    """Return all saved runs: metadata, summary metrics, per-question rows."""
    results_dir = Path(get_settings().eval_results_dir)
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            report = EvalReport.load(path)
        except (ValueError, KeyError, TypeError):
            continue  # skip a file that is not a saved run rather than fail the page
        runs.append({
            "id": path.stem,
            "meta": report.meta,
            "summary": report.summary(),
            "results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "answerable": r.answerable,
                    "hit": r.hit,
                    "correctness": r.correctness,
                    "mode": r.mode,
                    "tool_calls": r.tool_calls,
                    "tokens": r.input_tokens + r.output_tokens,
                    "latency_ms": r.latency_ms,
                    "answer_preview": r.generated_answer,
                }
                for r in report.results
            ],
        })
    # Runs carry an explicit `order` in their metadata so related runs
    # (e.g. the retrieval ladder) appear in a meaningful sequence.
    runs.sort(key=lambda run: (run["meta"].get("group", ""), run["meta"].get("order", 0), run["id"]))
    return {"runs": runs}
