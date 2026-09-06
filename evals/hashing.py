"""A deterministic embedder for the eval harness and the test suite.

Hashed bag-of-words vectors: no API key, no network, no PyTorch, and identical
output on every run. That makes the eval runnable in CI at zero cost and keeps
tests fast.

It is deliberately **not** a production embedding provider. It captures
exact-term overlap but not synonymy — essentially the signal BM25 already has —
so enabling it for real use would add cost and complexity for no recall. The
eval harness says as much in its output and treats its dense rows as a lower
bound.

This lives in one place because it was previously implemented twice, in the
harness and in the test fixtures, at two different dimensions — so the eval and
the tests were quietly measuring different things.
"""

from __future__ import annotations

import hashlib

import numpy as np

from pdfchat.lexical import tokenize
from pdfchat.models import Usage

DEFAULT_DIMENSIONS = 512


class HashingEmbedder:
    """Hashes word tokens into a fixed-width vector, then L2-normalises."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, *, name: str = "hash-bow"):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.name = name

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> tuple[np.ndarray, Usage]:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32), Usage()
        matrix = np.vstack([self._vector(t) for t in texts])
        return matrix, Usage(embedding_tokens=len(texts))

    def embed_query(self, text: str) -> tuple[np.ndarray, Usage]:
        return self._vector(text).reshape(1, -1), Usage(embedding_tokens=1)
