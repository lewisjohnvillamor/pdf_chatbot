"""The vector store interface every backend implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..models import Chunk


@dataclass(slots=True)
class ChunkRecord:
    """A chunk paired with its embedding, as handed to a store."""

    chunk: Chunk
    embedding: np.ndarray | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Persistence and similarity search over embedded chunks.

    Implementations must be safe to call with a zero-dimension embedding
    (embeddings disabled), in which case :meth:`search_dense` returns nothing
    and callers fall back to lexical retrieval.
    """

    dimensions: int

    def add(self, records: list[ChunkRecord], *, collection: str) -> int:
        """Insert or replace ``records``; returns the number written."""

    def delete_collection(self, collection: str) -> None:
        """Remove every chunk in ``collection``."""

    def list_documents(self, collection: str) -> list[tuple[str, str, int]]:
        """``(doc_id, filename, chunk_count)`` for everything in the collection."""

    def count(self, collection: str) -> int:
        """Number of stored chunks."""

    def all_chunks(self, collection: str, *, doc_ids: list[str] | None = None) -> list[Chunk]:
        """Every chunk, in insertion order — used to build the lexical index."""

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        collection: str,
        limit: int,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Cosine-similarity search; returns ``(chunk, score)`` best first."""

    def fetch_vectors(self, chunk_ids: list[str], *, collection: str) -> np.ndarray | None:
        """Embeddings for ``chunk_ids`` in that exact order.

        Returns ``None`` when any id is missing or the store holds no vectors,
        which tells callers to skip vector-space post-processing (MMR) rather
        than operate on partial data.
        """

    def close(self) -> None:
        """Release any held resources."""
