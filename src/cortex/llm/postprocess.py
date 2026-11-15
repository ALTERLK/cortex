"""Post-processing for LLM output: removing reasoning blocks.

Some reasoning models, served through an OpenAI-compatible endpoint, emit
their reasoning inline as <thinking>...</thinking> in the content. Users
should never see it — it is scratch work, not the answer.

The tag is not reliably spelled: the same model has been observed emitting
<thinking>, <antthinking>, and <anthinking>. The patterns below accept any
short alphabetic prefix before "thinking" instead of one exact string.

Two tools, one job:
  - strip_thinking():       for complete strings (non-streaming responses)
  - ThinkingStreamFilter:   for token streams, where a tag may arrive torn
                            across chunk boundaries ("<thi" + "nking>")
"""

from __future__ import annotations

import re

_NAME = r"[a-z]{0,4}thinking"
_OPEN = re.compile(rf"<{_NAME}>", re.IGNORECASE)
_CLOSE = re.compile(rf"</{_NAME}>", re.IGNORECASE)
_BLOCK = re.compile(rf"<{_NAME}>.*?</{_NAME}>\s*", re.IGNORECASE | re.DOTALL)
# An unfinished tag at the very end of a buffer: "<", "</", "<thi", "</antth"…
# Long enough for the longest tag accepted above, minus its closing ">".
_PARTIAL_TAG = re.compile(r"</?[a-z]{0,12}", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove all reasoning blocks from a complete string."""
    return _BLOCK.sub("", text).strip()


def _partial_tag_suffix(text: str) -> int:
    """Length of the trailing fragment of *text* that could still become a tag.

    This is what makes stream filtering correct. If the buffer ends in
    "<thi" it cannot be emitted yet — the next chunk might complete the tag.
    That fragment is held back and everything before it is emitted. Once
    more text arrives, the fragment either completes a tag or turns out to
    be ordinary text ("<this") and is released.
    """
    start = text.rfind("<")
    if start == -1 or not _PARTIAL_TAG.fullmatch(text, start):
        return 0
    return len(text) - start


class ThinkingStreamFilter:
    """Stateful filter that strips reasoning blocks from a stream of deltas.

    Usage::

        f = ThinkingStreamFilter()
        for delta in stream:
            visible = f.feed(delta)   # may be "" while inside a block
            ...
        visible = f.flush()           # emit anything still buffered
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        # After a closing tag, swallow following whitespace — it may arrive
        # in a later chunk, so the flag must survive across feed() calls.
        self._strip_ws = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []

        while True:
            if self._inside:
                close = _CLOSE.search(self._buf)
                if close is None:
                    # Still inside the block: discard the reasoning, but keep
                    # a tail that could be the start of the closing tag.
                    keep = _partial_tag_suffix(self._buf)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[close.end():]
                self._inside = False
                # Whitespace right after a block is formatting noise (the
                # model usually puts a newline there) — same as the \s* in
                # the whole-string pattern.
                self._strip_ws = True
            else:
                if self._strip_ws:
                    self._buf = self._buf.lstrip()
                    if not self._buf:
                        break  # all whitespace so far; keep waiting
                    self._strip_ws = False
                opening = _OPEN.search(self._buf)
                if opening is None:
                    emit_until = len(self._buf) - _partial_tag_suffix(self._buf)
                    out.append(self._buf[:emit_until])
                    self._buf = self._buf[emit_until:]
                    break
                out.append(self._buf[:opening.start()])
                self._buf = self._buf[opening.end():]
                self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """Emit whatever is still buffered at end of stream.

        A held-back partial tag that never completed was real text after all;
        an unterminated reasoning block is dropped entirely.
        """
        rest = "" if self._inside else self._buf
        self._buf = ""
        self._inside = False
        return rest
