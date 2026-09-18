from __future__ import annotations

import pytest

from evals.hashing import HashingEmbedder
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
        #: The cacheable prefix each call was given, so tests can assert that
        #: repeated calls reuse a byte-identical one.
        self.cache_prefixes: list[str | None] = []
        self._usage = Usage(input_tokens=100, output_tokens=50, cost_usd=0.001, calls=1)

    def _next(self) -> str:
        return self.responses.pop(0) if self.responses else "No response queued."

    def complete(self, system, user, *, max_tokens=None, cache_prefix=None):
        from pdfchat.llm import Completion

        self.prompts.append((system, user))
        self.cache_prefixes.append(cache_prefix)
        return Completion(text=self._next(), usage=self._usage, stop_reason="end_turn")

    def stream(self, system, user, *, max_tokens=None, cache_prefix=None):
        self.prompts.append((system, user))
        self.cache_prefixes.append(cache_prefix)
        for word in self._next().split(" "):
            yield word + " "

    def last_usage(self) -> Usage:
        return self._usage


@pytest.fixture()
def fake_model() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture()
def fake_embedder() -> HashingEmbedder:
    """The same embedder the eval harness uses, so tests and evals agree."""
    return HashingEmbedder(dimensions=256, name="fake-embedder")
