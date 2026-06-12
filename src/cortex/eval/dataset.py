"""QA dataset: load and represent evaluation question-answer pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAPair:
    """One evaluation example.

    Attributes:
        question:         The natural-language question.
        expected_answer:  Reference answer used by the LLM judge.
        expected_sources: Source labels that must appear in the top-k retrieved
                          chunks (relative paths, e.g. "fastapi/en/tutorial/cors.md").
                          A hit = ANY expected source retrieved.
        category:         Question type — factual | multihop | crosslingual |
                          followup | unanswerable. Lets the report show where
                          the system is weak, not just one blended number.
        answerable:       False for trap questions whose correct behaviour is
                          an honest refusal. Retrieval metrics are skipped.
        history:          Prior conversation turns (role/content dicts) for
                          follow-up questions that contain references like "it".
    """

    question: str
    expected_answer: str
    expected_sources: list[str]
    category: str = "factual"
    answerable: bool = True
    history: list[dict] = field(default_factory=list)


def load_dataset(path: Path | str = "data/eval_set.json") -> list[QAPair]:
    """Load the evaluation set from a JSON file.

    Required keys: question, expected_answer, expected_sources.
    Optional keys: category, answerable, history.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        QAPair(
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_sources=item["expected_sources"],
            category=item.get("category", "factual"),
            answerable=item.get("answerable", True),
            history=item.get("history", []),
        )
        for item in data
    ]
