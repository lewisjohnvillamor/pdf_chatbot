"""Postgres + pgvector store: persistent, shared, and query-pushdown capable.

Everything lives in one table so a chunk's text, provenance and embedding stay
consistent. Similarity search runs in the database (``<=>`` cosine distance),
so filtering by document is a WHERE clause rather than a client-side scan, and
the corpus never has to fit in the app's memory.

Schema and index creation are idempotent — :meth:`ensure_schema` is safe to
call on every start, which keeps deployment to "point it at a database".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import Settings
from ..errors import ConfigError, RetrievalError
from ..models import Chunk
from .base import ChunkRecord

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    id           bigserial PRIMARY KEY,
    collection   text        NOT NULL,
    chunk_id     text        NOT NULL,
    doc_id       text        NOT NULL,
    filename     text        NOT NULL,
    content      text        NOT NULL,
    page_start   integer     NOT NULL,
    page_end     integer     NOT NULL,
    ordinal      integer     NOT NULL,
    section      text,
    token_estimate integer   NOT NULL DEFAULT 0,
    embedding    vector({dimensions}),
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection, chunk_id)
);

CREATE INDEX IF NOT EXISTS {table}_collection_doc_idx
    ON {table} (collection, doc_id);
CREATE INDEX IF NOT EXISTS {table}_collection_ordinal_idx
    ON {table} (collection, ordinal);
-- Trigram index makes the SQL-side keyword prefilter cheap on large corpora.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS {table}_content_trgm_idx
    ON {table} USING gin (content gin_trgm_ops);
"""

VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
    ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists});
"""


def _from_pgvector(literal: str) -> np.ndarray:
    """Parse pgvector's ``[0.1,0.2,...]`` text form back into an array."""
    return np.fromstring(literal.strip()[1:-1], sep=",", dtype=np.float32)


def _to_pgvector(vector: np.ndarray) -> str:
    """pgvector accepts its literal form: ``[0.1,0.2,...]``."""
    return "[" + ",".join(f"{value:.7g}" for value in np.asarray(vector, dtype=np.float32)) + "]"


class PostgresVectorStore:
    """A :class:`~pdfchat.stores.base.VectorStore` backed by Postgres/pgvector."""

    def __init__(self, settings: Settings, *, dimensions: int):
        if not settings.database_url:
            raise ConfigError(
                "VECTOR_STORE=postgres requires DATABASE_URL, in the form "
                "postgresql://USER:PASSWORD@HOST:5432/DBNAME"
            )
        if dimensions <= 0:
            raise ConfigError(
                "VECTOR_STORE=postgres requires embeddings; set EMBEDDING_PROVIDER "
                "to 'openai' or 'local'."
            )
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise ConfigError(
                "VECTOR_STORE=postgres requires psycopg. Install with: "
                "pip install 'psycopg[binary,pool]'"
            ) from exc

        self.dimensions = dimensions
        self._table = settings.pg_table
        self._lists = settings.pg_ivfflat_lists
        # A pool keeps Streamlit's per-interaction reruns from opening a new
        # connection every time the user types.
        self._pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=settings.pg_pool_size,
            timeout=settings.request_timeout_s,
            open=True,
        )
        self.ensure_schema()

    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        """Create the table, indexes and extensions if they are missing."""
        try:
            with self._pool.connection() as conn:
                conn.execute(SCHEMA_SQL.format(table=self._table, dimensions=self.dimensions))
                conn.commit()
        except Exception as exc:
            raise ConfigError(
                f"Could not initialize the pgvector schema: {exc}. Confirm DATABASE_URL "
                "is reachable and the role may CREATE EXTENSION."
            ) from exc

    def ensure_vector_index(self) -> None:
        """Build the IVFFlat index once a collection is populated.

        pgvector's IVFFlat index must be created *after* rows exist, otherwise
        its centroids are trained on an empty table and recall collapses.
        """
        try:
            with self._pool.connection() as conn:
                conn.execute(VECTOR_INDEX_SQL.format(table=self._table, lists=self._lists))
                conn.commit()
        except Exception:
            # A missing ANN index only costs speed — exact scan still answers.
            logger.warning("pgvector_index_create_failed", exc_info=True)

    # ------------------------------------------------------------------
    def add(self, records: list[ChunkRecord], *, collection: str) -> int:
        if not records:
            return 0
        rows: list[tuple[Any, ...]] = []
        for record in records:
            chunk = record.chunk
            if record.embedding is None:
                continue
            rows.append(
                (
                    collection,
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.filename,
                    chunk.text,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.ordinal,
                    chunk.section,
                    chunk.token_estimate,
                    _to_pgvector(record.embedding),
                )
            )
        if not rows:
            return 0
        sql = f"""
            INSERT INTO {self._table}
                (collection, chunk_id, doc_id, filename, content, page_start, page_end,
                 ordinal, section, token_estimate, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (collection, chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                section = EXCLUDED.section
        """
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
                conn.commit()
        except Exception as exc:
            raise RetrievalError(f"Failed to write chunks to Postgres: {exc}") from exc
        self.ensure_vector_index()
        logger.info("pgvector_upsert", extra={"rows": len(rows), "collection": collection})
        return len(rows)

    def delete_collection(self, collection: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(f"DELETE FROM {self._table} WHERE collection = %s", (collection,))
            conn.commit()

    def count(self, collection: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT count(*) FROM {self._table} WHERE collection = %s", (collection,)
            ).fetchone()
        return int(row[0]) if row else 0

    def list_documents(self, collection: str) -> list[tuple[str, str, int]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT doc_id, filename, count(*) FROM {self._table}
                    WHERE collection = %s GROUP BY doc_id, filename ORDER BY filename""",
                (collection,),
            ).fetchall()
        return [(row[0], row[1], int(row[2])) for row in rows]

    def _row_to_chunk(self, row: tuple[Any, ...]) -> Chunk:
        return Chunk(
            chunk_id=row[0],
            doc_id=row[1],
            filename=row[2],
            text=row[3],
            page_start=row[4],
            page_end=row[5],
            ordinal=row[6],
            section=row[7],
            token_estimate=row[8] or 0,
        )

    _SELECT_COLUMNS = "chunk_id, doc_id, filename, content, page_start, page_end, ordinal, section, token_estimate"

    def all_chunks(self, collection: str, *, doc_ids: list[str] | None = None) -> list[Chunk]:
        sql = f"SELECT {self._SELECT_COLUMNS} FROM {self._table} WHERE collection = %s"
        params: list[Any] = [collection]
        if doc_ids:
            sql += " AND doc_id = ANY(%s)"
            params.append(list(doc_ids))
        sql += " ORDER BY filename, ordinal"
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def search_dense(
        self,
        query_vector: np.ndarray,
        *,
        collection: str,
        limit: int,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if limit <= 0:
            return []
        sql = f"""
            SELECT {self._SELECT_COLUMNS}, 1 - (embedding <=> %s::vector) AS similarity
            FROM {self._table}
            WHERE collection = %s AND embedding IS NOT NULL
        """
        params: list[Any] = [_to_pgvector(query_vector), collection]
        if doc_ids:
            sql += " AND doc_id = ANY(%s)"
            params.append(list(doc_ids))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([_to_pgvector(query_vector), limit])
        try:
            with self._pool.connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            raise RetrievalError(f"pgvector search failed: {exc}") from exc
        return [(self._row_to_chunk(row), float(row[9])) for row in rows]

    def fetch_vectors(self, chunk_ids: list[str], *, collection: str) -> np.ndarray | None:
        if not chunk_ids:
            return None
        sql = f"""
            SELECT chunk_id, embedding::text FROM {self._table}
            WHERE collection = %s AND chunk_id = ANY(%s) AND embedding IS NOT NULL
        """
        try:
            with self._pool.connection() as conn:
                rows = conn.execute(sql, (collection, list(chunk_ids))).fetchall()
        except Exception:
            logger.warning("pgvector_fetch_vectors_failed", exc_info=True)
            return None
        found = {row[0]: _from_pgvector(row[1]) for row in rows}
        if len(found) != len(set(chunk_ids)):
            return None
        return np.vstack([found[chunk_id] for chunk_id in chunk_ids])

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception:  # pragma: no cover - shutdown best effort
            logger.debug("pgvector_pool_close_failed", exc_info=True)
