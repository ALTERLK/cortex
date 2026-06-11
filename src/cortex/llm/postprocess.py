"""Post-processing for LLM output: removing <thinking> blocks.

Extended-thinking models (e.g. claude-*-thinking via an OpenAI-compatible
proxy) emit their reasoning inside <thinking>...</thinking> tags in the
content. Users should never see those — they are scratch work, not answer.

Two tools, one job:
  - strip_thinking():       for complete strings (non-streaming responses)
  - ThinkingStreamFilter:   for token streams, where a tag may arrive torn
                            across chunk boundaries ("<thi" + "nking>")
"""

from __future__ import annotations

import re

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", flags=re.DOTALL)

_OPEN = "<thinking>"
_CLOSE = "</thinking>"


def strip_thinking(text: str) -> str:
    """Remove all <thinking>...</thinking> blocks from a complete string."""
    return _THINKING_RE.sub("", text).strip()


def _partial_tag_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of *text* that is a proper prefix of *tag*.

    NOTE (learning): this is the heart of stream filtering. If the buffer ends
    in "<thi", we cannot emit it yet — the next chunk might complete the tag.
    We hold back exactly that many characters and emit the rest.
    """
    max_len = min(len(text), len(tag) - 1)
    for k in range(max_len, 0, -1):
        if tag.startswith(text[-k:]):
            return k
    return 0


class ThinkingStreamFilter:
    """Stateful filter that strips <thinking> blocks from a stream of deltas.

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
                idx = self._buf.find(_CLOSE)
                if idx == -1:
                    # Still inside the block: discard thinking content, but
                    # keep a tail that could be the start of the closing tag.
                    keep = _partial_tag_suffix(self._buf, _CLOSE)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[idx + len(_CLOSE):]
                self._inside = False
                # Whitespace right after a block is formatting noise (the
                # model usually puts a newline there) — same as the \s* in
                # strip_thinking's regex.
                self._strip_ws = True
            else:
                if self._strip_ws:
                    self._buf = self._buf.lstrip()
                    if not self._buf:
                        break  # all whitespace so far; keep waiting
                    self._strip_ws = False
                idx = self._buf.find(_OPEN)
                if idx == -1:
                    keep = _partial_tag_suffix(self._buf, _OPEN)
                    emit_until = len(self._buf) - keep
                    out.append(self._buf[:emit_until])
                    self._buf = self._buf[emit_until:]
                    break
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(_OPEN):]
                self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """Emit whatever is still buffered at end of stream.

        A held-back partial tag that never completed was real text after all;
        an unterminated <thinking> block is dropped entirely.
        """
        rest = "" if self._inside else self._buf
        self._buf = ""
        self._inside = False
        return rest
