"""Hybrid retrieval: BM25 + dense vectors, fused and diversified.

Three stages, each fixing a specific failure of the naive
``vectorstore.as_retriever()`` approach:

1. **Recall** — run lexical and dense search independently. Dense search finds
   paraphrases; BM25 finds exact terms (``Article 7(b)``, ``NADPH``) that
   embeddings blur away.
2. **Fusion** — combine the two ranked lists with Reciprocal Rank Fusion. RRF
   works on ranks, not raw scores, so it needs no score calibration between
   two backends whose scales are unrelated.
3. **Diversity** — apply Maximal Marginal Relevance so the final context is not
   six near-copies of the same paragraph, which is what wastes the context
   window and makes answers repeat themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .embeddings import Embedder
from .lexical import BM25, tokenize
from .models import Chunk, ScoredChunk, Usage
from .stores.base import VectorStore

logger = logging.getLogger(__name__)

#: RRF damping constant. 60 is the value from the original Cormack et al.
#: paper and is insensitive enough that tuning it rarely pays off.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], weights: list[float] | None = None, *, k: int = RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists into one score map: ``sum(weight / (k + rank))``."""
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights, strict=True):
        for rank, identifier in enumerate(ids, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    return fused


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    k: int,
    lambda_mult: float = 0.6,
) -> list[int]:
    """Select ``k`` candidate indices balancing relevance against redundancy."""
    if candidate_vectors.size == 0:
        return []
    k = min(k, len(candidate_vectors))
    relevance = candidate_vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1)
    selected: list[int] = [int(np.argmax(relevance))]
    while len(selected) < k:
        # Penalize each remaining candidate by its similarity to what we already took.
        redundancy = np.max(candidate_vectors @ candidate_vectors[selected].T, axis=1)
        score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
        score[selected] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        selected.append(best)
    return selected


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    usage: Usage
    lexical_hits: int = 0
    dense_hits: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class HybridRetriever:
    """Query-time orchestration over a vector store plus an in-process BM25 index.

    The BM25 index is rebuilt whenever the store's chunk count changes, which
    is cheap (linear in corpus size) and keeps lexical recall correct after
    new uploads without a separate invalidation protocol.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings):
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._bm25: BM25 | None = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_signature: tuple[str, int] | None = None

    # ------------------------------------------------------------------
    def _ensure_lexical_index(self, collection: str) -> None:
        signature = (collection, self._store.count(collection))
        if self._bm25_signature == signature and self._bm25 is not None:
            return
        chunks = self._store.all_chunks(collection)
        self._bm25_chunks = chunks
        self._bm25 = BM25([tokenize(chunk.text) for chunk in chunks]) if chunks else None
        self._bm25_signature = signature
        logger.info("bm25_index_built", extra={"chunks": len(chunks), "collection": collection})

    def invalidate(self) -> None:
        """Force a lexical rebuild on the next search."""
        self._bm25_signature = None

    # ------------------------------------------------------------------
    def retrieve(
        self, question: str, *, collection: str, doc_ids: list[str] | None = None
    ) -> RetrievalResult:
        """Return the best passages for ``question``, best first."""
        settings = self._settings
        question = question.strip()
        if not question:
            return RetrievalResult(chunks=[], usage=Usage())

        self._ensure_lexical_index(collection)
        by_id: dict[str, Chunk] = {}
        lexical_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        usage = Usage()

        # --- lexical ---------------------------------------------------
        lexical_ids: list[str] = []
        if self._bm25 is not None:
            wanted = set(doc_ids) if doc_ids else None
            for index, score in self._bm25.top_n(question, settings.candidate_k * 2):
                chunk = self._bm25_chunks[index]
                if wanted is not None and chunk.doc_id not in wanted:
                    continue
                by_id[chunk.chunk_id] = chunk
                lexical_scores[chunk.chunk_id] = score
                lexical_ids.append(chunk.chunk_id)
                if len(lexical_ids) >= settings.candidate_k:
                    break

        # --- dense -----------------------------------------------------
        dense_ids: list[str] = []
        query_vector: np.ndarray | None = None
        if self._embedder.dimensions > 0:
            vectors, embed_usage = self._embedder.embed_query(question)
            usage = usage.add(embed_usage)
            query_vector = vectors[0]
            hits = self._store.search_dense(
                query_vector, collection=collection, limit=settings.candidate_k, doc_ids=doc_ids
            )
            for chunk, score in hits:
                by_id[chunk.chunk_id] = chunk
                dense_scores[chunk.chunk_id] = score
                dense_ids.append(chunk.chunk_id)

        if not by_id:
            return RetrievalResult(chunks=[], usage=usage)

        # --- fusion ----------------------------------------------------
        dense_weight = settings.hybrid_dense_weight
        fused = reciprocal_rank_fusion(
            [dense_ids, lexical_ids], weights=[dense_weight, 1.0 - dense_weight]
        )
        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        candidate_ids = [chunk_id for chunk_id, _ in ordered][: settings.candidate_k]

        # --- diversity -------------------------------------------------
        if query_vector is not None and len(candidate_ids) > settings.top_k:
            candidate_vectors = self._vectors_for(candidate_ids, collection)
            if candidate_vectors is not None:
                order = maximal_marginal_relevance(
                    query_vector,
                    candidate_vectors,
                    k=settings.top_k,
                    lambda_mult=settings.mmr_lambda,
                )
                candidate_ids = [candidate_ids[i] for i in order]

        selected = candidate_ids[: settings.top_k]
        results = [
            ScoredChunk(
                chunk=by_id[chunk_id],
                score=fused.get(chunk_id, 0.0),
                dense_score=dense_scores.get(chunk_id, 0.0),
                lexical_score=lexical_scores.get(chunk_id, 0.0),
                rank=position + 1,
            )
            for position, chunk_id in enumerate(selected)
            if chunk_id in by_id
        ]
        logger.info(
            "retrieval_complete",
            extra={
                "question_chars": len(question),
                "lexical_hits": len(lexical_ids),
                "dense_hits": len(dense_ids),
                "returned": len(results),
            },
        )
        return RetrievalResult(
            chunks=results,
            usage=usage,
            lexical_hits=len(lexical_ids),
            dense_hits=len(dense_ids),
        )

    def _vectors_for(self, chunk_ids: list[str], collection: str) -> np.ndarray | None:
        """Embeddings for the fused candidates, or None if the store can't supply them."""
        return self._store.fetch_vectors(chunk_ids, collection=collection)
