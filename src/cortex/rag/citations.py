"""Answer checks that need no LLM call: citations and refusals.

`check_citations` verifies that the [N] markers in an answer point at real
sources. It cannot tell whether a cited passage actually supports the
sentence — that needs a judge — but it does catch two cheap, common
failures: a citation number that matches no retrieved passage, and an
answer that makes claims with no citations at all.

`is_refusal` recognises the "not in the documents" answer the prompts
mandate. Together they give a free signal for "this answer did not work",
which is what lets auto mode escalate to the agent only when needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Phrases that signal an honest refusal. The system prompts mandate the
# first one, in English whatever the language of the question, precisely so
# that a refusal can be detected by string matching; the others catch close
# paraphrases.
_REFUSAL_MARKERS = (
    "don't have information",
    "do not have information",
    "no information about",
    "not covered in the provided documents",
    # Models sometimes refuse in the question's language despite the
    # instruction. Chinese questions are common here, so the usual Chinese
    # phrasings are recognised too; a refusal must not be scored as a failure
    # because of the language it was written in.
    "没有找到",
    "未找到",
    "找不到",
    "没有关于",
    "没有相关",
)

# A refusal opens the answer. Looking only at the opening keeps an answer
# that says "the documents do not cover X" halfway through a real answer
# from being mistaken for one.
_REFUSAL_WINDOW = 200


def is_refusal(answer: str) -> bool:
    """True when *answer* opens by saying the documents do not contain the answer."""
    opening = answer[:_REFUSAL_WINDOW].lower()
    return any(marker in opening for marker in _REFUSAL_MARKERS)


_CITATION = re.compile(r"\[(\d{1,3})\]")
# Code is skipped so that `items[0]` or `[1]` inside a snippet is not
# mistaken for a citation.
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


@dataclass(frozen=True)
class CitationCheck:
    """Result of checking one answer against its source list."""

    cited: list[int]    # distinct valid citation numbers, in order of first use
    invalid: list[int]  # distinct numbers that match no source

    @property
    def grounded(self) -> bool:
        """True when the answer cites at least one source and no phantom ones."""
        return bool(self.cited) and not self.invalid

    def as_dict(self) -> dict:
        return {"cited": self.cited, "invalid": self.invalid, "grounded": self.grounded}


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    """Classify every [N] in *answer* as valid (1..n_sources) or invalid."""
    cited: list[int] = []
    invalid: list[int] = []
    for match in _CITATION.finditer(_CODE.sub(" ", answer)):
        number = int(match.group(1))
        bucket = cited if 1 <= number <= n_sources else invalid
        if number not in bucket:
            bucket.append(number)
    return CitationCheck(cited=cited, invalid=invalid)
