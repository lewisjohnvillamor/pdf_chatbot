"""Tests for the reranking stage.

A reranker that silently drops passages or reorders them wrongly is worse than
none, because the pipeline trusts its output over fusion order.
"""

from __future__ import annotations

import pytest

from pdfchat.config import Settings
from pdfchat.errors import ConfigError
from pdfchat.models import ScoredChunk
from pdfchat.reranking import (
    LLMReranker,
    NullReranker,
    Reranker,
    build_reranker,
)
from pdfchat.retrieval import HybridRetriever
from pdfchat.stores.base import ChunkRecord
from pdfchat.stores.memory import MemoryVectorStore
from tests.conftest import FakeChatModel, make_chunk

TEXTS = [
    "Article 7(a) requires submissions to be made in electronic form.",
    "Article 7(b) states that late filing incurs a penalty of 250 euros per month.",
    "Article 7(c) permits an extension of up to 30 days in exceptional circumstances.",
    "Fines are applied progressively depending on the severity of the breach.",
]


def candidates() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=make_chunk(f"d:{i:05d}", text, page=i + 1), score=1.0 / (i + 1))
        for i, text in enumerate(TEXTS)
    ]


class ReverseReranker:
    """A deterministic stand-in that reverses order — proves wiring, not quality."""

    name = "reverse"

    def rerank(self, query, cands, *, top_k):
        from pdfchat.models import Usage

        flipped = list(reversed(cands))[:top_k]
        return [
            ScoredChunk(
                chunk=c.chunk,
                score=float(len(flipped) - i),
                dense_score=c.dense_score,
                lexical_score=c.lexical_score,
                rank=i + 1,
            )
            for i, c in enumerate(flipped)
        ], Usage()


# --- protocol / factory ----------------------------------------------------
def test_null_reranker_preserves_order_and_truncates():
    result, usage = NullReranker().rerank("q", candidates(), top_k=2)
    assert [c.chunk.chunk_id for c in result] == ["d:00000", "d:00001"]
    assert usage.calls == 0


def test_null_reranker_handles_no_candidates():
    result, usage = NullReranker().rerank("q", [], top_k=5)
    assert result == []
    assert usage.calls == 0


def test_build_reranker_defaults_to_none():
    assert isinstance(build_reranker(Settings()), NullReranker)


def test_build_reranker_rejects_unknown_backend():
    with pytest.raises(ConfigError, match="Unknown RERANKER"):
        build_reranker(Settings(reranker="magic"))  # type: ignore[arg-type]


def test_llm_reranker_requires_a_model():
    with pytest.raises(ConfigError, match="requires a chat model"):
        build_reranker(Settings(reranker="llm", rerank_candidates=20))


def test_config_rejects_unknown_reranker():
    with pytest.raises(ConfigError, match="Unknown RERANKER"):
        Settings(reranker="nope").validated()  # type: ignore[arg-type]


def test_config_requires_enough_candidates_to_rerank():
    with pytest.raises(ConfigError, match="RERANK_CANDIDATES"):
        Settings(reranker="llm", top_k=10, rerank_candidates=5).validated()


def test_rerankers_satisfy_the_protocol():
    assert isinstance(NullReranker(), Reranker)
    assert isinstance(LLMReranker(FakeChatModel()), Reranker)


# --- LLM reranker ----------------------------------------------------------
def test_llm_reranker_orders_by_returned_scores():
    model = FakeChatModel(['{"scores": {"0": 1, "1": 9, "2": 3, "3": 0}}'])
    result, usage = LLMReranker(model).rerank("penalty?", candidates(), top_k=3)
    assert [c.chunk.chunk_id for c in result] == ["d:00001", "d:00002", "d:00000"]
    assert result[0].rank == 1
    assert usage.calls == 1


def test_llm_reranker_preserves_component_scores_for_the_ui():
    original = candidates()
    original[1].dense_score, original[1].lexical_score = 0.87, 4.2
    model = FakeChatModel(['{"scores": {"0": 1, "1": 9, "2": 3, "3": 0}}'])
    result, _ = LLMReranker(model).rerank("q", original, top_k=1)
    assert result[0].dense_score == 0.87
    assert result[0].lexical_score == 4.2


def test_llm_reranker_falls_back_to_fusion_order_on_bad_json():
    model = FakeChatModel(["not json"])
    result, _ = LLMReranker(model).rerank("q", candidates(), top_k=2)
    assert [c.chunk.chunk_id for c in result] == ["d:00000", "d:00001"]


def test_llm_reranker_falls_back_when_the_model_raises():
    class Exploding(FakeChatModel):
        def complete(self, system, user, *, max_tokens=None):
            raise RuntimeError("provider down")

    result, _ = LLMReranker(Exploding()).rerank("q", candidates(), top_k=2)
    assert len(result) == 2, "a failed rerank must never return zero passages"


def test_llm_reranker_handles_partial_scores():
    # The model omitted indices 2 and 3; they must rank last, not vanish.
    model = FakeChatModel(['{"scores": {"0": 2, "1": 8}}'])
    result, _ = LLMReranker(model).rerank("q", candidates(), top_k=4)
    assert [c.chunk.chunk_id for c in result[:2]] == ["d:00001", "d:00000"]
    assert len(result) == 4


def test_llm_reranker_handles_no_candidates():
    result, usage = LLMReranker(FakeChatModel()).rerank("q", [], top_k=5)
    assert result == [] and usage.calls == 0


# --- integration with the retriever ---------------------------------------
@pytest.fixture()
def store(fake_embedder) -> MemoryVectorStore:
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
    vectors, _ = fake_embedder.embed_documents(TEXTS)
    store.add(
        [
            ChunkRecord(chunk=make_chunk(f"d:{i:05d}", t, page=i + 1), embedding=vectors[i])
            for i, t in enumerate(TEXTS)
        ],
        collection="c",
    )
    return store


def test_retriever_applies_the_reranker(settings, fake_embedder, store):
    tuned = settings.with_overrides(top_k=4, candidate_k=10, rerank_candidates=10)
    baseline = HybridRetriever(store, fake_embedder, tuned).retrieve("article", collection="c")
    reranked = HybridRetriever(store, fake_embedder, tuned, reranker=ReverseReranker()).retrieve(
        "article", collection="c"
    )

    base_ids = [c.chunk.chunk_id for c in baseline.chunks]
    new_ids = [c.chunk.chunk_id for c in reranked.chunks]
    assert set(base_ids) == set(new_ids), "reranking must not add or drop passages"
    assert new_ids != base_ids, "the reranker's order must actually be applied"


def test_retriever_without_a_reranker_is_unchanged(settings, fake_embedder, store):
    plain = HybridRetriever(store, fake_embedder, settings).retrieve("article", collection="c")
    explicit_null = HybridRetriever(
        store, fake_embedder, settings, reranker=NullReranker()
    ).retrieve("article", collection="c")
    assert [c.chunk.chunk_id for c in plain.chunks] == [
        c.chunk.chunk_id for c in explicit_null.chunks
    ]


def test_reranker_usage_is_accumulated(settings, fake_embedder, store):
    tuned = settings.with_overrides(rerank_candidates=10)
    model = FakeChatModel(['{"scores": {"0": 5, "1": 9}}'])
    result = HybridRetriever(store, fake_embedder, tuned, reranker=LLMReranker(model)).retrieve(
        "article", collection="c"
    )
    assert result.usage.calls >= 1, "reranking cost must reach the session total"
