"""Conversation persistence.

A self-hosted deployment should not lose a study session when the browser tab
reloads or the container restarts. Conversations are written to Postgres when
one is configured, and held in memory otherwise, behind one interface so the
UI never has to care which is active.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from .config import Settings
from .models import Answer, Citation

logger = logging.getLogger(__name__)

HISTORY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id            bigserial PRIMARY KEY,
    conversation_id text      NOT NULL,
    collection    text        NOT NULL,
    role          text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content       text        NOT NULL,
    citations     jsonb       NOT NULL DEFAULT '[]'::jsonb,
    verdict       text,
    cost_usd      double precision NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS {table}_conversation_idx
    ON {table} (conversation_id, id);
"""


@dataclass(slots=True)
class Turn:
    """One message in a conversation."""

    role: str
    content: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    verdict: str | None = None
    cost_usd: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class HistoryStore(Protocol):
    def append(self, conversation_id: str, collection: str, turn: Turn) -> None: ...

    def load(self, conversation_id: str) -> list[Turn]: ...

    def clear(self, conversation_id: str) -> None: ...


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def turn_from_answer(answer: Answer) -> Turn:
    """Build the assistant turn that corresponds to a generated answer."""
    return Turn(
        role="assistant",
        content=answer.text,
        citations=[_citation_dict(c) for c in answer.citations],
        verdict=answer.verdict,
        cost_usd=answer.usage.cost_usd,
    )


def _citation_dict(citation: Citation) -> dict[str, Any]:
    return {
        "marker": citation.marker,
        "filename": citation.filename,
        "page_start": citation.page_start,
        "page_end": citation.page_end,
        "chunk_id": citation.chunk_id,
    }


class MemoryHistoryStore:
    """Process-local history. Lost on restart; fine for single-session use."""

    def __init__(self) -> None:
        self._turns: dict[str, list[Turn]] = {}

    def append(self, conversation_id: str, collection: str, turn: Turn) -> None:
        self._turns.setdefault(conversation_id, []).append(turn)

    def load(self, conversation_id: str) -> list[Turn]:
        return list(self._turns.get(conversation_id, []))

    def clear(self, conversation_id: str) -> None:
        self._turns.pop(conversation_id, None)


class PostgresHistoryStore:
    """Durable conversation history in Postgres."""

    def __init__(self, settings: Settings):
        from psycopg_pool import ConnectionPool

        self._table = f"{settings.pg_table}_messages"
        self._pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=max(2, settings.pg_pool_size // 2),
            open=True,
        )
        with self._pool.connection() as conn:
            conn.execute(HISTORY_SCHEMA_SQL.format(table=self._table))
            conn.commit()

    def append(self, conversation_id: str, collection: str, turn: Turn) -> None:
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    f"""INSERT INTO {self._table}
                        (conversation_id, collection, role, content, citations, verdict, cost_usd)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)""",
                    (
                        conversation_id,
                        collection,
                        turn.role,
                        turn.content,
                        json.dumps(turn.citations),
                        turn.verdict,
                        turn.cost_usd,
                    ),
                )
                conn.commit()
        except Exception:
            # History is a convenience; never fail a user's answer over it.
            logger.warning("history_write_failed", exc_info=True)

    def load(self, conversation_id: str) -> list[Turn]:
        try:
            with self._pool.connection() as conn:
                rows = conn.execute(
                    f"""SELECT role, content, citations, verdict, cost_usd, created_at
                        FROM {self._table} WHERE conversation_id = %s ORDER BY id""",
                    (conversation_id,),
                ).fetchall()
        except Exception:
            logger.warning("history_read_failed", exc_info=True)
            return []
        return [
            Turn(
                role=row[0],
                content=row[1],
                citations=row[2] or [],
                verdict=row[3],
                cost_usd=float(row[4] or 0.0),
                created_at=row[5],
            )
            for row in rows
        ]

    def clear(self, conversation_id: str) -> None:
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    f"DELETE FROM {self._table} WHERE conversation_id = %s", (conversation_id,)
                )
                conn.commit()
        except Exception:
            logger.warning("history_clear_failed", exc_info=True)


def build_history_store(settings: Settings) -> HistoryStore:
    """Pick the durable store when Postgres is configured, else in-memory."""
    if settings.persist_conversations and settings.database_url:
        try:
            return PostgresHistoryStore(settings)
        except Exception:
            logger.warning("history_store_fallback_to_memory", exc_info=True)
    return MemoryHistoryStore()
