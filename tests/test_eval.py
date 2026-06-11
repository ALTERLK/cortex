"""Tests for the evaluation metrics and dataset loader.

All offline — no LLM calls, no Qdrant, no embedding model.
"""

from __future__ import annotations

import json

import pytest

from cortex.eval.dataset import load_dataset
from cortex.eval.metrics import (
    _parse_judge_response,
    compute_retrieval_hit,
    hit_rate,
    mrr,
)
from cortex.eval.runner import EvalReport, EvalResult
from cortex.ingest.store import SearchResult

# ---------------------------------------------------------------------------
# Metric unit tests (pure functions)
# ---------------------------------------------------------------------------


def test_hit_rate_all_hits() -> None:
    assert hit_rate([True, True, True]) == pytest.approx(1.0)


def test_hit_rate_no_hits() -> None:
    assert hit_rate([False, False]) == pytest.approx(0.0)


def test_hit_rate_partial() -> None:
    assert hit_rate([True, False, True, False]) == pytest.approx(0.5)


def test_hit_rate_empty() -> None:
    assert hit_rate([]) == 0.0


def test_mrr_first_rank() -> None:
    assert mrr([1.0]) == pytest.approx(1.0)


def test_mrr_second_rank() -> None:
    assert mrr([0.5]) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert mrr([0.0]) == pytest.approx(0.0)


def test_mrr_mixed() -> None:
    # 1/1 + 1/2 = 1.5 → mean = 0.75
    assert mrr([1.0, 0.5]) == pytest.approx(0.75)


def test_mrr_empty() -> None:
    assert mrr([]) == 0.0


# ---------------------------------------------------------------------------
# compute_retrieval_hit
# ---------------------------------------------------------------------------


def make_result(source: str, score: float = 0.9) -> SearchResult:
    return SearchResult(score=score, text="x", source=source, chunk_index=0)


def test_hit_first_position() -> None:
    results = [make_result("target.md"), make_result("other.md")]
    hit, rr = compute_retrieval_hit(results, ["target.md"], top_k=5)
    assert hit is True
    assert rr == pytest.approx(1.0)


def test_hit_second_position() -> None:
    results = [make_result("other.md"), make_result("target.md")]
    hit, rr = compute_retrieval_hit(results, ["target.md"], top_k=5)
    assert hit is True
    assert rr == pytest.approx(0.5)


def test_no_hit_outside_top_k() -> None:
    results = [make_result("other.md")] * 3 + [make_result("target.md")]
    hit, rr = compute_retrieval_hit(results, ["target.md"], top_k=3)
    assert hit is False
    assert rr == pytest.approx(0.0)


def test_no_hit_empty_results() -> None:
    hit, rr = compute_retrieval_hit([], ["target.md"], top_k=5)
    assert hit is False
    assert rr == 0.0


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------


def test_parse_clean_json() -> None:
    scores = _parse_judge_response('{"correctness": 4, "faithfulness": 3}')
    assert scores == {"correctness": 4.0, "faithfulness": 3.0}


def test_parse_json_with_surrounding_text() -> None:
    scores = _parse_judge_response('Here is my evaluation:\n{"correctness": 5, "faithfulness": 4}\nDone.')
    assert scores["correctness"] == 5.0


def test_parse_json_with_thinking_block() -> None:
    text = "<thinking>Let me evaluate this carefully.</thinking>\n{\"correctness\": 3, \"faithfulness\": 4}"
    scores = _parse_judge_response(text)
    assert scores["correctness"] == 3.0


def test_parse_invalid_json_returns_default() -> None:
    scores = _parse_judge_response("I cannot score this.")
    assert scores["correctness"] == 2.5
    assert scores["faithfulness"] == 2.5


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def test_load_dataset(tmp_path: pytest.TempPathFactory) -> None:
    data = [
        {
            "question": "What is RAG?",
            "expected_answer": "RAG augments LLMs.",
            "expected_sources": ["sample.md"],
        }
    ]
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    pairs = load_dataset(f)
    assert len(pairs) == 1
    assert pairs[0].question == "What is RAG?"
    assert pairs[0].expected_sources == ["sample.md"]


# ---------------------------------------------------------------------------
# EvalReport aggregation
# ---------------------------------------------------------------------------


def make_eval_result(hit: bool, rr: float, correctness: float, faithfulness: float) -> EvalResult:
    return EvalResult(
        question="Q?",
        hit=hit,
        reciprocal_rank=rr,
        generated_answer="A.",
        correctness=correctness,
        faithfulness=faithfulness,
    )


def test_report_hit_rate() -> None:
    report = EvalReport([
        make_eval_result(True, 1.0, 4.0, 4.0),
        make_eval_result(False, 0.0, 2.0, 3.0),
    ])
    assert report.hit_rate == pytest.approx(0.5)


def test_report_mrr() -> None:
    report = EvalReport([
        make_eval_result(True, 1.0, 4.0, 4.0),
        make_eval_result(True, 0.5, 4.0, 4.0),
    ])
    assert report.mrr == pytest.approx(0.75)


def test_report_avg_correctness() -> None:
    report = EvalReport([
        make_eval_result(True, 1.0, 3.0, 4.0),
        make_eval_result(True, 1.0, 5.0, 4.0),
    ])
    assert report.avg_correctness == pytest.approx(4.0)
