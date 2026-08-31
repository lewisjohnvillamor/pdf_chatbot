from __future__ import annotations

import numpy as np
import pytest

from pdfchat.config import Settings
from pdfchat.models import Chunk, Document, Page, Usage


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        chat_provider="anthropic",
        anthropic_api_key="test-key",
        embedding_provider="none",
        chunk_size=400,
        chunk_overlap=60,
        top_k=3,
        candidate_k=10,
        cache_dir=None,
    ).validated()


@pytest.fixture()
def document() -> Document:
    pages = [
        Page(
            number=1,
            text=(
                "PHOTOSYNTHESIS\n\nPhotosynthesis converts light energy into chemical "
                "energy stored as glucose. It takes place in the chloroplasts of plant "
                "cells and requires water and carbon dioxide."
            ),
            char_count=190,
        ),
        Page(
            number=2,
            text=(
                "CELLULAR RESPIRATION\n\nCellular respiration releases the energy stored "
                "in glucose. In eukaryotes it happens in the mitochondria and yields "
                "roughly 30 to 32 molecules of ATP per glucose molecule."
            ),
            char_count=195,
        ),
    ]
    return Document(
        doc_id="doc1",
        filename="biology.pdf",
        pages=pages,
        sha256="a" * 64,
        size_bytes=4096,
    )


def make_chunk(chunk_id: str, text: str, *, page: int = 1, doc_id: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        filename="biology.pdf",
        text=text,
        page_start=page,
        page_end=page,
        ordinal=int(chunk_id.split(":")[-1]) if ":" in chunk_id else 0,
    )


class FakeChatModel:
    """A scripted chat model: returns queued responses, records prompts."""

    provider = "fake"
    name = "fake-model"

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.prompts: list[tuple[str, str]] = []
        self._usage = Usage(input_tokens=100, output_tokens=50, cost_usd=0.001, calls=1)

    def _next(self) -> str:
        return self.responses.pop(0) if self.responses else "No response queued."

    def complete(self, system, user, *, max_tokens=None):
        from pdfchat.llm import Completion

        self.prompts.append((system, user))
        return Completion(text=self._next(), usage=self._usage, stop_reason="end_turn")

    def stream(self, system, user, *, max_tokens=None):
        self.prompts.append((system, user))
        for word in self._next().split(" "):
            yield word + " "

    def last_usage(self) -> Usage:
        return self._usage


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings.

    Word-level rather than character-level: texts sharing vocabulary land close
    together, so similarity in tests means what it means in production.
    """

    name = "fake-embedder"
    dimensions = 256

    def _vector(self, text: str) -> np.ndarray:
        import hashlib

        from pdfchat.lexical import tokenize

        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32), Usage()
        return np.vstack([self._vector(t) for t in texts]), Usage(embedding_tokens=len(texts))

    def embed_query(self, text):
        return self._vector(text).reshape(1, -1), Usage(embedding_tokens=1)


@pytest.fixture()
def fake_model() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
