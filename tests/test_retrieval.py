from __future__ import annotations

import numpy as np

from pdfchat.retrieval import (
    HybridRetriever,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from pdfchat.stores.base import ChunkRecord
from pdfchat.stores.memory import MemoryVectorStore
from tests.conftest import make_chunk

TEXTS = [
    "Photosynthesis converts light energy into glucose inside the chloroplasts.",
    "Cellular respiration releases energy from glucose inside the mitochondria.",
    "The cell membrane is a phospholipid bilayer regulating transport.",
    "DNA replication is semiconservative and requires DNA polymerase.",
    "Enzymes lower activation energy and are highly substrate specific.",
]


def build_store(embedder) -> MemoryVectorStore:
    store = MemoryVectorStore(dimensions=embedder.dimensions)
    chunks = [make_chunk(f"doc1:{i:05d}", text, page=i + 1) for i, text in enumerate(TEXTS)]
    vectors, _ = embedder.embed_documents(TEXTS)
    store.add(
        [ChunkRecord(chunk=c, embedding=vectors[i]) for i, c in enumerate(chunks)],
        collection="c",
    )
    return store


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    # "b" is 2nd and 1st; "a" is 1st and 2nd — both beat single-list entries.
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]


def test_rrf_respects_weights():
    dense_only = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 0.0])
    assert dense_only["a"] > dense_only["b"]


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == {}


def _unit(rows: list[list[float]]) -> np.ndarray:
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


# Index 0 and 1 are near-duplicates of each other. Index 2 is slightly *less*
# relevant but points elsewhere, so relevance-only ranking picks 1 and a
# diversity-aware ranker picks 2 — which is exactly what MMR must trade off.
_QUERY = np.array([1.0, 0.0], dtype=np.float32)
_CANDIDATES = _unit([[0.90, 0.44], [0.89, 0.46], [0.86, -0.51]])


def test_mmr_prefers_diverse_results():
    selected = maximal_marginal_relevance(_QUERY, _CANDIDATES, k=2, lambda_mult=0.5)
    assert selected[0] == 0
    assert selected[1] == 2, "MMR kept the near-duplicate instead of the diverse passage"


def test_mmr_pure_relevance_when_lambda_is_one():
    # With no diversity penalty, the ranking is relevance order: 0, then 1.
    assert maximal_marginal_relevance(_QUERY, _CANDIDATES, k=2, lambda_mult=1.0) == [0, 1]


def test_mmr_handles_empty_candidates():
    assert maximal_marginal_relevance(np.array([1.0]), np.zeros((0, 1)), k=3) == []


def test_hybrid_retrieval_finds_the_right_passage(settings, fake_embedder):
    store = build_store(fake_embedder)
    retriever = HybridRetriever(store, fake_embedder, settings)
    result = retriever.retrieve("mitochondria respiration", collection="c")
    assert not result.is_empty
    assert "mitochondria" in result.chunks[0].chunk.text.lower()
    # Both retrievers must contribute, or "hybrid" is a misnomer.
    assert result.lexical_hits > 0
    assert result.dense_hits > 0


def test_retrieval_respects_document_filter(settings, fake_embedder):
    store = build_store(fake_embedder)
    store.add(
        [
            ChunkRecord(
                chunk=make_chunk("doc2:00000", "Mitochondria appear here too.", doc_id="doc2"),
                embedding=fake_embedder.embed_documents(["Mitochondria appear here too."])[0][0],
            )
        ],
        collection="c",
    )
    retriever = HybridRetriever(store, fake_embedder, settings)
    result = retriever.retrieve("mitochondria", collection="c", doc_ids=["doc2"])
    assert result.chunks
    assert {scored.chunk.doc_id for scored in result.chunks} == {"doc2"}


def test_retrieval_works_without_embeddings(settings):
    from pdfchat.embeddings import NullEmbedder

    embedder = NullEmbedder()
    store = MemoryVectorStore(dimensions=0)
    store.add(
        [ChunkRecord(chunk=make_chunk(f"doc1:{i:05d}", t)) for i, t in enumerate(TEXTS)],
        collection="c",
    )
    retriever = HybridRetriever(store, embedder, settings)
    result = retriever.retrieve("enzymes activation energy", collection="c")
    assert result.dense_hits == 0, "no embedder means no dense candidates"
    assert result.lexical_hits > 0, "BM25 must carry retrieval on its own"
    assert "enzymes" in result.chunks[0].chunk.text.lower()


def test_retrieval_returns_nothing_for_empty_query(settings, fake_embedder):
    retriever = HybridRetriever(build_store(fake_embedder), fake_embedder, settings)
    assert retriever.retrieve("   ", collection="c").is_empty


def test_retrieval_returns_nothing_for_empty_collection(settings, fake_embedder):
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
    retriever = HybridRetriever(store, fake_embedder, settings)
    assert retriever.retrieve("anything", collection="empty").is_empty


def test_retrieval_honours_top_k(settings, fake_embedder):
    store = build_store(fake_embedder)
    retriever = HybridRetriever(store, fake_embedder, settings.with_overrides(top_k=2))
    assert len(retriever.retrieve("energy glucose", collection="c").chunks) <= 2


def test_new_documents_are_visible_after_invalidate(settings, fake_embedder):
    store = build_store(fake_embedder)
    retriever = HybridRetriever(store, fake_embedder, settings)
    retriever.retrieve("glucose", collection="c")  # builds the lexical index
    text = "Osmosis moves water across a semipermeable membrane."
    store.add(
        [
            ChunkRecord(
                chunk=make_chunk("doc1:00099", text),
                embedding=fake_embedder.embed_documents([text])[0][0],
            )
        ],
        collection="c",
    )
    retriever.invalidate()
    result = retriever.retrieve("osmosis semipermeable", collection="c")
    assert "osmosis" in result.chunks[0].chunk.text.lower()


def test_mmr_uses_supplied_relevance_over_embedding_similarity():
    """The reranker's judgement must survive diversification.

    Without this, MMR recomputes relevance from the query embedding and
    silently discards whatever the (expensive) reranker decided.
    """
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = _unit([[1.0, 0.02], [0.0, 1.0]])
    # Embedding similarity strongly prefers index 0; an upstream reranker says
    # the opposite. With lambda=1.0 (pure relevance) the reranker must win.
    order = maximal_marginal_relevance(
        query, candidates, k=2, lambda_mult=1.0, relevance=np.array([0.1, 0.9])
    )
    assert order[0] == 1, "MMR ignored the supplied relevance scores"


def test_mmr_falls_back_to_embedding_relevance_when_none_supplied():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = _unit([[1.0, 0.02], [0.0, 1.0]])
    assert maximal_marginal_relevance(query, candidates, k=2, lambda_mult=1.0)[0] == 0


def test_mmr_rejects_mismatched_relevance_length():
    import pytest

    with pytest.raises(ValueError, match="one score per candidate"):
        maximal_marginal_relevance(
            np.array([1.0, 0.0], dtype=np.float32),
            _unit([[1.0, 0.0], [0.0, 1.0]]),
            k=2,
            relevance=np.array([1.0]),
        )


def test_mmr_handles_uniform_relevance_without_dividing_by_zero():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = _unit([[1.0, 0.0], [0.0, 1.0]])
    order = maximal_marginal_relevance(
        query, candidates, k=2, lambda_mult=0.5, relevance=np.array([0.5, 0.5])
    )
    assert sorted(order) == [0, 1]


def test_retrieve_many_embeds_every_query_in_one_request(settings, fake_embedder):
    """The study tools probe with several queries; that must be one round trip."""

    class CountingEmbedder:
        name = "counting"

        def __init__(self, inner):
            self._inner = inner
            self.dimensions = inner.dimensions
            self.document_calls = 0
            self.query_calls = 0

        def embed_documents(self, texts):
            self.document_calls += 1
            return self._inner.embed_documents(texts)

        def embed_query(self, text):
            self.query_calls += 1
            return self._inner.embed_query(text)

    counting = CountingEmbedder(fake_embedder)
    store = build_store(fake_embedder)
    retriever = HybridRetriever(store, counting, settings)

    results = retriever.retrieve_many(
        ["glucose energy", "enzymes catalysts", "membrane transport"], collection="c"
    )
    assert len(results) == 3
    assert counting.document_calls == 1, "queries were not batched"
    assert counting.query_calls == 0, "a per-query embedding call slipped through"


def test_retrieve_many_matches_retrieve_one_by_one(settings, fake_embedder):
    store = build_store(fake_embedder)
    retriever = HybridRetriever(store, fake_embedder, settings)
    queries = ["glucose energy", "enzymes catalysts"]
    batched = retriever.retrieve_many(queries, collection="c")
    single = [retriever.retrieve(q, collection="c") for q in queries]
    for b, s in zip(batched, single, strict=True):
        assert [c.chunk.chunk_id for c in b.chunks] == [c.chunk.chunk_id for c in s.chunks]


def test_retrieve_many_handles_empty_and_blank_queries(settings, fake_embedder):
    retriever = HybridRetriever(build_store(fake_embedder), fake_embedder, settings)
    assert retriever.retrieve_many([], collection="c") == []
    assert retriever.retrieve_many(["   ", ""], collection="c") == []


def test_retrieve_many_reports_the_embedding_cost_once(settings, fake_embedder):
    retriever = HybridRetriever(build_store(fake_embedder), fake_embedder, settings)
    results = retriever.retrieve_many(["glucose", "enzymes", "membrane"], collection="c")
    total = sum(r.usage.embedding_tokens for r in results)
    assert total == 3, "the batched embedding cost must be counted, and only once"
