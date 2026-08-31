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
    assert result.dense_hits == 0
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
