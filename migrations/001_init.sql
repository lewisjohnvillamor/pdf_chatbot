-- Baseline schema for the pgvector store.
--
-- The application also creates these objects at startup (ensure_schema), so
-- this file is mainly for provisioning a database ahead of time or for review
-- by a DBA. It is idempotent and safe to re-run.
--
-- NOTE: the embedding dimension must match your embedding model:
--   text-embedding-3-small -> 1536   (the default below)
--   text-embedding-3-large -> 3072
--   all-MiniLM-L6-v2       -> 384
-- Change the vector(1536) below if you use a different model.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS pdfchat_chunks (
    id             bigserial PRIMARY KEY,
    collection     text        NOT NULL,
    chunk_id       text        NOT NULL,
    doc_id         text        NOT NULL,
    filename       text        NOT NULL,
    content        text        NOT NULL,
    page_start     integer     NOT NULL,
    page_end       integer     NOT NULL,
    ordinal        integer     NOT NULL,
    section        text,
    token_estimate integer     NOT NULL DEFAULT 0,
    embedding      vector(1536),
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection, chunk_id)
);

CREATE INDEX IF NOT EXISTS pdfchat_chunks_collection_doc_idx
    ON pdfchat_chunks (collection, doc_id);
CREATE INDEX IF NOT EXISTS pdfchat_chunks_collection_ordinal_idx
    ON pdfchat_chunks (collection, ordinal);
CREATE INDEX IF NOT EXISTS pdfchat_chunks_content_trgm_idx
    ON pdfchat_chunks USING gin (content gin_trgm_ops);

-- The IVFFlat index is created by the application after rows exist: building it
-- on an empty table trains its centroids on nothing and destroys recall.

CREATE TABLE IF NOT EXISTS pdfchat_chunks_messages (
    id              bigserial PRIMARY KEY,
    conversation_id text        NOT NULL,
    collection      text        NOT NULL,
    role            text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content         text        NOT NULL,
    citations       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    verdict         text,
    cost_usd        double precision NOT NULL DEFAULT 0,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pdfchat_chunks_messages_conversation_idx
    ON pdfchat_chunks_messages (conversation_id, id);
