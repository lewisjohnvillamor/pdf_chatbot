"""Lexical retrieval (BM25 Okapi) implemented in-process.

Dense embeddings miss exact tokens — an equation name, a statute number, a
rare acronym — because those carry little semantic signal. BM25 catches them.
Fusing both (see :mod:`pdfchat.retrieval`) is measurably better than either
alone, which is why this is worth ~80 lines rather than a heavyweight
dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")

#: Structured references that the plain word tokenizer destroys.
#:
#: "Article 7(c)" splits into "article", "7" and "c"; the last two are single
#: characters and get dropped, so 7(a), 7(b) and 7(c) all reduce to "article"
#: and become indistinguishable to keyword search. That is precisely the case
#: hybrid retrieval is supposed to win - a rare exact identifier an embedding
#: blurs away - so the identifier is emitted whole, as one atomic token.
_IDENTIFIER = re.compile(
    r"""
      \d+ \s* \( \s* [a-z0-9]{1,3} \s* \)   # 7(c), 12 (a)
    | § \s* \d+ (?: \.\d+ )*                  # section 3, 3.1
    | \b \d+ (?: \.\d+ )+ \b                 # 3.1, 12.4.2
    """,
    re.VERBOSE,
)
_SPACES = re.compile(r"\s+")

#: Very common words carry no discriminative signal and inflate scoring cost.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "our",
        "we",
        "us",
        "do",
        "does",
        "did",
        "not",
        "no",
        "than",
        "then",
        "there",
        "these",
        "those",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus any structured identifiers, stopwords removed.

    Identifiers are emitted in addition to the ordinary words, not instead of
    them, so "Article 7(c)" still matches on "article" while "7(c)" supplies
    the discriminator that tells it from 7(a) and 7(b).
    """
    lowered = text.lower()
    identifiers = [_SPACES.sub("", m.group(0)) for m in _IDENTIFIER.finditer(lowered)]
    words = [t for t in _TOKEN.findall(lowered) if t not in STOPWORDS and len(t) > 1]
    return identifiers + words


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    __slots__ = ("_avg_len", "_doc_freqs", "_doc_lens", "_idf", "_n", "b", "k1")

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self._doc_lens = [len(doc) for doc in corpus]
        self._n = len(corpus)
        self._avg_len = (sum(self._doc_lens) / self._n) if self._n else 0.0

        containing: Counter[str] = Counter()
        for freqs in self._doc_freqs:
            containing.update(freqs.keys())
        # Robertson/Sparck-Jones IDF with the +1 smoothing that keeps it positive.
        self._idf = {
            term: math.log(1 + (self._n - count + 0.5) / (count + 0.5))
            for term, count in containing.items()
        }

    def __len__(self) -> int:
        return self._n

    def scores(self, query: str) -> list[float]:
        """BM25 score of every corpus document against ``query``."""
        terms = tokenize(query)
        results = [0.0] * self._n
        if not terms or not self._n or self._avg_len == 0:
            return results
        for index, freqs in enumerate(self._doc_freqs):
            length = self._doc_lens[index]
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
            total = 0.0
            for term in terms:
                frequency = freqs.get(term)
                if not frequency:
                    continue
                total += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / (frequency + norm)
            results[index] = total
        return results

    def top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        """The ``n`` highest-scoring documents as ``(index, score)``, best first."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:n]
