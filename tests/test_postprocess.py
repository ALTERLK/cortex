"""Tests for <thinking> block stripping — both whole-string and streaming."""

from __future__ import annotations

from cortex.llm.postprocess import ThinkingStreamFilter, strip_thinking

# ---------------------------------------------------------------------------
# strip_thinking (complete strings)
# ---------------------------------------------------------------------------


def test_strip_removes_block() -> None:
    assert strip_thinking("<thinking>hmm</thinking>Answer.") == "Answer."


def test_strip_removes_multiple_blocks() -> None:
    text = "<thinking>a</thinking>One. <thinking>b</thinking>Two."
    assert strip_thinking(text) == "One. Two."


def test_strip_passthrough_without_tags() -> None:
    assert strip_thinking("Plain answer [1].") == "Plain answer [1]."


def test_strip_multiline_block() -> None:
    text = "<thinking>line one\nline two</thinking>\nAnswer."
    assert strip_thinking(text) == "Answer."


def test_strip_accepts_tag_spelling_variants() -> None:
    # The same model has emitted each of these spellings.
    for name in ("thinking", "antthinking", "anthinking", "Thinking"):
        assert strip_thinking(f"<{name}>plan</{name}>\nAnswer [1].") == "Answer [1]."


def test_strip_leaves_other_tags_alone() -> None:
    text = "Use the <code>thinking</code> flag and <b>bold</b>."
    assert strip_thinking(text) == text


# ---------------------------------------------------------------------------
# ThinkingStreamFilter (token streams)
# ---------------------------------------------------------------------------


def run_filter(chunks: list[str]) -> str:
    f = ThinkingStreamFilter()
    out = "".join(f.feed(c) for c in chunks)
    return out + f.flush()


def test_filter_passthrough() -> None:
    assert run_filter(["Hello ", "world."]) == "Hello world."


def test_filter_whole_block_in_one_chunk() -> None:
    assert run_filter(["<thinking>x</thinking>Answer."]) == "Answer."


def test_filter_tag_torn_across_chunks() -> None:
    # The opening tag arrives split as "<thi" + "nking>".
    assert run_filter(["<thi", "nking>secret</thi", "nking>Visible."]) == "Visible."


def test_filter_one_char_chunks() -> None:
    text = "<thinking>abc</thinking>Done."
    assert run_filter(list(text)) == "Done."


def test_filter_text_before_and_after_block() -> None:
    assert run_filter(["Pre. <thinking>x</thinking>", "Post."]) == "Pre. Post."


def test_filter_false_alarm_partial_tag() -> None:
    # "<th" looks like a tag start but turns out to be plain text;
    # flush() must release it.
    assert run_filter(["a < b and ", "<th", "is is fine"]) == "a < b and <this is fine"


def test_filter_handles_tag_spelling_variants() -> None:
    assert run_filter(["<anthi", "nking>secret</anth", "inking>\nVisible."]) == "Visible."
    assert run_filter(list("<antthinking>abc</antthinking>Done.")) == "Done."


def test_filter_passes_ordinary_markup_through() -> None:
    # Angle brackets that are not reasoning tags must come out unchanged,
    # including when a tag is split across chunks.
    assert run_filter(["List<int> x; a <", "b> c </", "b> if a < b"]) == "List<int> x; a <b> c </b> if a < b"


def test_filter_trailing_angle_bracket_is_released_on_flush() -> None:
    assert run_filter(["value <"]) == "value <"


def test_filter_unterminated_block_dropped() -> None:
    assert run_filter(["Answer. <thinking>never closed"]) == "Answer. "


def test_filter_strips_whitespace_after_block() -> None:
    assert run_filter(["<thinking>x</thinking>\n\n  Answer."]) == "Answer."


def test_filter_strips_whitespace_arriving_in_later_chunk() -> None:
    # Closing tag and the following newline arrive in separate deltas.
    assert run_filter(["<thinking>x</thinking>", "\n", "\n", "Answer."]) == "Answer."
