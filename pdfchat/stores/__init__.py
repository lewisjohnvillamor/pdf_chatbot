"""Pluggable vector stores.

``memory`` keeps everything in a NumPy matrix with an on-disk cache — perfect
for a single-user session. ``postgres`` persists chunks and embeddings in
Postgres with the pgvector extension, so a corpus survives restarts, is shared
across app replicas, and scales past what fits in one process's RAM.
"""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import ChunkRecord, VectorStore
from .memory import MemoryVectorStore

__all__ = [
    "ChunkRecord",
    "MemoryVectorStore",
    "VectorStore",
    "build_store",
]


def build_store(settings: Settings, *, dimensions: int) -> VectorStore:
    """Instantiate the store named by ``VECTOR_STORE``."""
    backend = settings.vector_store
    if backend == "memory":
        return MemoryVectorStore(dimensions=dimensions, cache_dir=settings.cache_dir)
    if backend == "postgres":
        from .postgres import PostgresVectorStore  # imported lazily: psycopg is optional

        return PostgresVectorStore(settings, dimensions=dimensions)
    raise ConfigError(f"Unknown VECTOR_STORE {backend!r}; expected 'memory' or 'postgres'.")
