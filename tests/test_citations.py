"""Tests for the deterministic citation check."""

from __future__ import annotations

from cortex.rag.citations import check_citations, strip_citations


def test_valid_citations_are_collected_in_order_of_first_use() -> None:
    check = check_citations("B is true [2]. A too [1][2]. Also C [3].", n_sources=3)
    assert check.cited == [2, 1, 3]
    assert check.invalid == []
    assert check.grounded


def test_citation_beyond_source_count_is_invalid() -> None:
    check = check_citations("Claim [1]. Invented [7].", n_sources=2)
    assert check.cited == [1]
    assert check.invalid == [7]
    assert not check.grounded


def test_zero_is_not_a_citation_number() -> None:
    assert check_citations("Claim [0].", n_sources=3).invalid == [0]


def test_answer_without_citations_is_not_grounded() -> None:
    check = check_citations("A confident claim with no source.", n_sources=5)
    assert check.cited == [] and check.invalid == []
    assert not check.grounded


def test_no_sources_means_every_citation_is_invalid() -> None:
    assert check_citations("Claim [1].", n_sources=0).invalid == [1]


def test_brackets_inside_code_are_ignored() -> None:
    answer = "Use the first item [1]:\n```python\nvalue = items[0]\nrow = matrix[9]\n```\nor `args[4]`."
    check = check_citations(answer, n_sources=2)
    assert check.cited == [1]
    assert check.invalid == []


def test_duplicates_are_reported_once() -> None:
    check = check_citations("[9] and again [9], plus [1] and [1].", n_sources=1)
    assert check.cited == [1]
    assert check.invalid == [9]


def test_as_dict_shape() -> None:
    assert check_citations("Yes [1].", 1).as_dict() == {"cited": [1], "invalid": [], "grounded": True}


def test_strip_citations_removes_markers_and_the_space_before_them() -> None:
    text = "Runs after the response [1][3]. Also this [2]."
    assert strip_citations(text) == "Runs after the response. Also this."


def test_strip_citations_leaves_code_alone() -> None:
    text = "Use `items[0]` here [1].\n```\nx[2] = 1\n```"
    assert strip_citations(text) == "Use `items[0]` here.\n```\nx[2] = 1\n```"
