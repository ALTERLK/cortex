"""QA dataset: load and represent evaluation question-answer pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QAPair:
    """One evaluation example.

    Attributes:
        question:         The natural-language question.
        expected_answer:  Reference answer used by the LLM judge.
        expected_sources: Filenames that must appear in the top-k retrieved chunks.
    """

    question: str
    expected_answer: str
    expected_sources: list[str]


def load_dataset(path: Path | str = "data/eval_set.json") -> list[QAPair]:
    """Load the evaluation set from a JSON file.

    The JSON must be a list of objects with keys:
        question, expected_answer, expected_sources
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        QAPair(
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_sources=item["expected_sources"],
        )
        for item in data
    ]
