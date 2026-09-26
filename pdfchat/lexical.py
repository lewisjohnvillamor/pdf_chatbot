"""Lexical retrieval (BM25 Okapi) implemented in-process.

Dense embeddings miss exact tokens — an equation name, a statute number, a
rare acronym — because those carry little semantic signal. BM25 catches them.
Fusing both (see :mod:`pdfchat.retrieval`) is measurably better than either
alone, which is why this is worth ~80 lines rather than a heavyweight
dependency.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

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
    """Okapi BM25 over a fixed corpus, served from an inverted index.

    The obvious implementation scores every document against every query. That
    is O(N) per query no matter how rare the terms are, and it dominates as the
    corpus grows - 45 ms per query over 32k passages, paid again for each of
    the five probes a study-tool run issues.

    An inverted index inverts the loop: for each query term, walk only the
    documents that actually contain it. Cost becomes proportional to the length
    of those postings lists, which for the rare, discriminative terms that
    decide BM25 rankings is a tiny fraction of the corpus. Scores are
    unchanged - this is the same formula, visited in a different order.
    """

    __slots__ = ("_avg_len", "_doc_lens", "_idf", "_n", "_norms", "_postings", "b", "k1")

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_lens = [len(doc) for doc in corpus]
        self._n = len(corpus)
        self._avg_len = (sum(self._doc_lens) / self._n) if self._n else 0.0

        # term -> [(document index, term frequency), ...]
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, doc in enumerate(corpus):
            for term, frequency in Counter(doc).items():
                postings[term].append((index, frequency))
        self._postings = dict(postings)

        # Robertson/Sparck-Jones IDF with the +1 smoothing that keeps it positive.
        self._idf = {
            term: math.log(1 + (self._n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self._postings.items()
        }

        # The length normalisation depends only on the document, so it is
        # computed once here rather than once per (document, query term) pair.
        if self._avg_len:
            self._norms = [
                self.k1 * (1 - self.b + self.b * length / self._avg_len)
                for length in self._doc_lens
            ]
        else:
            self._norms = [0.0] * self._n

    def __len__(self) -> int:
        return self._n

    def _accumulate(self, query: str) -> dict[int, float]:
        """Partial scores for the documents any query term actually reaches."""
        totals: dict[int, float] = {}
        for term in tokenize(query):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = self._idf[term]
            weight = idf * (self.k1 + 1)
            for index, frequency in plist:
                contribution = weight * frequency / (frequency + self._norms[index])
                totals[index] = totals.get(index, 0.0) + contribution
        return totals

    def scores(self, query: str) -> list[float]:
        """BM25 score of every corpus document against ``query``."""
        results = [0.0] * self._n
        for index, score in self._accumulate(query).items():
            results[index] = score
        return results

    def top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        """The ``n`` highest-scoring documents as ``(index, score)``, best first.

        Selection is a bounded heap rather than a full sort: O(m log n) over the
        m documents the query touched, instead of O(N log N) over the corpus.
        """
        totals = self._accumulate(query)
        if not totals:
            return []
        return heapq.nlargest(n, totals.items(), key=lambda pair: pair[1])
