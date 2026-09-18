"""In-process vector store backed by a NumPy matrix, with an on-disk cache.

Embeddings are the expensive part of ingestion, so a collection is persisted
under ``CACHE_DIR`` keyed by a fingerprint of its documents and settings.
Re-uploading the same PDFs then costs nothing and returns instantly.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..models import Chunk
from .base import ChunkRecord

logger = logging.getLogger(__name__)


class MemoryVectorStore:
    """Exact cosine search over an in-memory float32 matrix.

    Exact search is the right default here: brute force over a few thousand
    chunks costs well under a millisecond, and it avoids the recall loss and
    tuning burden of an approximate index at this scale.
    """

    def __init__(self, *, dimensions: int, cache_dir: Path | None = None):
        self.dimensions = dimensions
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = threading.RLock()
        self._chunks: dict[str, list[Chunk]] = {}
        self._matrix: dict[str, np.ndarray] = {}
        # Embeddings arrive in batches during indexing. Appending them to one
        # array copies the whole thing every time, which is quadratic: 20k
        # chunks spent ~2.8s purely re-copying. Pending blocks are held here
        # and concatenated once, on the first read that needs them.
        self._pending: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    def add(self, records: list[ChunkRecord], *, collection: str) -> int:
        if not records:
            return 0
        with self._lock:
            chunks = self._chunks.setdefault(collection, [])
            existing = {chunk.chunk_id for chunk in chunks}
            fresh = [r for r in records if r.chunk.chunk_id not in existing]
            if not fresh:
                return 0
            chunks.extend(record.chunk for record in fresh)

            if self.dimensions > 0:
                new_rows = np.vstack(
                    [
                        record.embedding
                        if record.embedding is not None
                        else np.zeros(self.dimensions, dtype=np.float32)
                        for record in fresh
                    ]
                ).astype(np.float32)
                self._pending.setdefault(collection, []).append(new_rows)
            return len(fresh)

    def _materialize(self, collection: str) -> np.ndarray | None:
        """Fold any pending blocks into the collection's matrix. Caller holds the lock."""
        pending = self._pending.pop(collection, None)
        if pending:
            current = self._matrix.get(collection)
            blocks = ([current] if current is not None else []) + pending
            self._matrix[collection] = np.vstack(blocks)
        return self._matrix.get(collection)

    def delete_collection(self, collection: str) -> None:
        with self._lock:
            self._chunks.pop(collection, None)
            self._matrix.pop(collection, None)
            self._pending.pop(collection, None)

    def count(self, collection: str) -> int:
        return len(self._chunks.get(collection, []))

    def list_documents(self, collection: str) -> list[tuple[str, str, int]]:
        counts: dict[tuple[str, str], int] = {}
        for chunk in self._chunks.get(collection, []):
            counts[(chunk.doc_id, chunk.filename)] = (
                counts.get((chunk.doc_id, chunk.filename), 0) + 1
            )
        return [(doc_id, name, count) for (doc_id, name), count in counts.items()]

    def all_chunks(self, collection: str, *, doc_ids: list[str] | None = None) -> list[Chunk]:
        chunks = self._chunks.get(collection, [])
        if doc_ids is None:
            return list(chunks)
        wanted = set(doc_ids)
        return [chunk for chunk in chunks if chunk.doc_id in wanted]

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        collection: str,
        limit: int,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self.dimensions == 0 or limit <= 0:
            return []
        with self._lock:
            matrix = self._materialize(collection)
            chunks = self._chunks.get(collection, [])
        if matrix is None or not len(chunks):
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        # Both sides are L2-normalized upstream, so a dot product is the cosine.
        similarities = matrix @ query

        if doc_ids is not None:
            wanted = set(doc_ids)
            mask = np.array([chunk.doc_id in wanted for chunk in chunks], dtype=bool)
            if not mask.any():
                return []
            similarities = np.where(mask, similarities, -np.inf)

        take = min(limit, len(chunks))
        top = np.argpartition(-similarities, take - 1)[:take]
        top = top[np.argsort(-similarities[top])]
        return [(chunks[i], float(similarities[i])) for i in top if np.isfinite(similarities[i])]

    def fetch_vectors(self, chunk_ids: list[str], *, collection: str) -> np.ndarray | None:
        if self.dimensions == 0 or not chunk_ids:
            return None
        with self._lock:
            matrix = self._materialize(collection)
            chunks = self._chunks.get(collection, [])
        if matrix is None or not chunks:
            return None
        positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        try:
            rows = [positions[chunk_id] for chunk_id in chunk_ids]
        except KeyError:
            return None
        return matrix[rows]

    def close(self) -> None:  # nothing to release
        return None

    # --- persistence ---------------------------------------------------
    def _cache_path(self, fingerprint: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{fingerprint}.npz"

    def save(self, collection: str, fingerprint: str) -> bool:
        """Persist a collection under ``fingerprint``. Returns True on success."""
        path = self._cache_path(fingerprint)
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            chunks = self._chunks.get(collection, [])
            matrix = self._materialize(collection)
            np.savez_compressed(
                path,
                chunks=json.dumps([asdict(chunk) for chunk in chunks]),
                matrix=matrix if matrix is not None else np.zeros((0, 0), dtype=np.float32),
                dimensions=np.int32(self.dimensions),
            )
            return True
        except OSError:
            # A read-only or full disk must never break the running session.
            logger.warning("index_cache_write_failed", extra={"path": str(path)}, exc_info=True)
            return False

    def load(self, collection: str, fingerprint: str) -> bool:
        """Restore a previously saved collection. Returns True if it was found."""
        path = self._cache_path(fingerprint)
        if path is None or not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as payload:
                dimensions = int(payload["dimensions"])
                if dimensions != self.dimensions:
                    return False  # embedding model changed; rebuild
                chunks = [Chunk(**record) for record in json.loads(str(payload["chunks"]))]
                matrix = payload["matrix"].astype(np.float32)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("index_cache_read_failed", extra={"path": str(path)}, exc_info=True)
            return False
        with self._lock:
            self._chunks[collection] = chunks
            if matrix.size:
                self._matrix[collection] = matrix
        logger.info("index_cache_hit", extra={"chunks": len(chunks), "fingerprint": fingerprint})
        return True
