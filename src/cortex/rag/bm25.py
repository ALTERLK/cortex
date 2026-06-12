"""Hand-written BM25 sparse index.

Why hand-written instead of a library (rank_bm25, fastembed, Qdrant sparse
vectors)? This project's rule is that every retrieval component must be
explainable line by line — BM25 is ~80 lines and is THE classic lexical
ranking function, so it earns its place as readable code.

NOTE (learning): BM25 scores a document for a query as

    sum over query terms t of:
        IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

where tf = term frequency in the doc, dl = doc length, avgdl = average doc
length. k1 (~1.5) controls term-frequency saturation: the 2nd occurrence of
a word matters less than the 1st. b (~0.75) controls length normalisation:
long documents get penalised so they can't win just by containing everything.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_LATIN_TOKEN = re.compile(r"[a-z0-9_]+")
_HAS_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer for mixed Chinese/English text.

    Latin words are extracted by regex; runs containing CJK characters are
    segmented with jieba (lazy import — pure-Latin corpora never load it).
    """
    text = text.lower()
    if _HAS_CJK.search(text):
        import jieba

        tokens: list[str] = []
        for tok in jieba.cut(text):
            tok = tok.strip()
            if not tok:
                continue
            if _HAS_CJK.search(tok):
                tokens.append(tok)
            else:
                tokens.extend(_LATIN_TOKEN.findall(tok))
        return tokens
    return _LATIN_TOKEN.findall(text)


@dataclass(frozen=True)
class BM25Hit:
    """One scored document from the sparse index."""

    doc_id: int   # position in the corpus passed to fit()
    score: float


class BM25Index:
    """In-memory BM25 over a list of documents.

    Built once at startup from the chunk texts already stored in Qdrant —
    no second persistent index to keep in sync.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._doc_freqs: list[Counter[str]] = []   # term -> tf, per document
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def fit(self, texts: list[str]) -> None:
        """Index *texts*; doc_id in results = position in this list."""
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter[str] = Counter()  # term -> number of docs containing it

        for text in texts:
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            df.update(freqs.keys())

        n = len(texts)
        self._avgdl = (sum(self._doc_lens) / n) if n else 0.0
        # Robertson/Sparck-Jones IDF with +1 smoothing (never negative).
        self._idf = {
            term: math.log(1 + (n - n_t + 0.5) / (n_t + 0.5))
            for term, n_t in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[BM25Hit]:
        """Score every document for *query*, return the top_k by BM25."""
        if not self._doc_freqs:
            return []

        query_terms = [t for t in tokenize(query) if t in self._idf]
        if not query_terms:
            return []

        scores = [0.0] * len(self._doc_freqs)
        for term in query_terms:
            idf = self._idf[term]
            for doc_id, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[doc_id]
                denom = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                scores[doc_id] += idf * tf * (self._k1 + 1) / denom

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [BM25Hit(doc_id=i, score=scores[i]) for i in ranked[:top_k] if scores[i] > 0]
