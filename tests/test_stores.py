from __future__ import annotations

import numpy as np
import pytest

from pdfchat.config import Settings
from pdfchat.errors import ConfigError
from pdfchat.stores import build_store
from pdfchat.stores.base import ChunkRecord
from pdfchat.stores.memory import MemoryVectorStore
from tests.conftest import make_chunk


@pytest.fixture()
def populated(fake_embedder) -> MemoryVectorStore:
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
    texts = ["alpha content here", "beta content here", "gamma content here"]
    vectors, _ = fake_embedder.embed_documents(texts)
    store.add(
        [
            ChunkRecord(chunk=make_chunk(f"doc1:{i:05d}", t, page=i + 1), embedding=vectors[i])
            for i, t in enumerate(texts)
        ],
        collection="c",
    )
    return store


def test_add_and_count(populated):
    assert populated.count("c") == 3
    assert populated.count("other") == 0


def test_add_is_idempotent_on_chunk_id(populated, fake_embedder):
    vectors, _ = fake_embedder.embed_documents(["alpha content here"])
    added = populated.add(
        [ChunkRecord(chunk=make_chunk("doc1:00000", "alpha content here"), embedding=vectors[0])],
        collection="c",
    )
    assert added == 0
    assert populated.count("c") == 3


def test_list_documents_groups_by_file(populated):
    assert populated.list_documents("c") == [("doc1", "biology.pdf", 3)]


def test_search_dense_orders_by_similarity(populated, fake_embedder):
    query, _ = fake_embedder.embed_query("gamma content here")
    hits = populated.search_dense(query[0], collection="c", limit=3)
    assert hits[0][0].text.startswith("gamma")
    assert hits[0][1] >= hits[1][1]


def test_search_dense_filters_by_document(populated, fake_embedder):
    query, _ = fake_embedder.embed_query("alpha")
    assert populated.search_dense(query[0], collection="c", limit=3, doc_ids=["missing"]) == []


def test_fetch_vectors_preserves_requested_order(populated):
    ids = ["doc1:00002", "doc1:00000"]
    vectors = populated.fetch_vectors(ids, collection="c")
    assert vectors is not None and vectors.shape == (2, populated.dimensions)


def test_fetch_vectors_returns_none_when_an_id_is_missing(populated):
    assert populated.fetch_vectors(["doc1:00000", "nope"], collection="c") is None


def test_delete_collection_removes_everything(populated):
    populated.delete_collection("c")
    assert populated.count("c") == 0
    assert populated.list_documents("c") == []


def test_cache_roundtrip_restores_chunks_and_vectors(tmp_path, fake_embedder):
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions, cache_dir=tmp_path)
    texts = ["first passage", "second passage"]
    vectors, _ = fake_embedder.embed_documents(texts)
    store.add(
        [
            ChunkRecord(chunk=make_chunk(f"doc1:{i:05d}", t), embedding=vectors[i])
            for i, t in enumerate(texts)
        ],
        collection="c",
    )
    assert store.save("c", "fingerprint1")

    restored = MemoryVectorStore(dimensions=fake_embedder.dimensions, cache_dir=tmp_path)
    assert restored.load("c", "fingerprint1")
    assert restored.count("c") == 2
    query, _ = fake_embedder.embed_query("second passage")
    assert restored.search_dense(query[0], collection="c", limit=1)[0][0].text == "second passage"


def test_cache_miss_on_unknown_fingerprint(tmp_path, fake_embedder):
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions, cache_dir=tmp_path)
    assert not store.load("c", "never-saved")


def test_cache_rejects_a_dimension_mismatch(tmp_path, fake_embedder):
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions, cache_dir=tmp_path)
    store.add(
        [
            ChunkRecord(
                chunk=make_chunk("doc1:00000", "x"),
                embedding=np.zeros(fake_embedder.dimensions, dtype=np.float32),
            )
        ],
        collection="c",
    )
    store.save("c", "fp")
    # A different embedding model must never reuse the cached vectors.
    other = MemoryVectorStore(dimensions=fake_embedder.dimensions + 1, cache_dir=tmp_path)
    assert not other.load("c", "fp")


def test_build_store_rejects_unknown_backend():
    with pytest.raises(ConfigError, match="Unknown VECTOR_STORE"):
        build_store(Settings(vector_store="redis"), dimensions=8)  # type: ignore[arg-type]


def test_postgres_settings_require_a_database_url():
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        Settings(vector_store="postgres", embedding_provider="openai").validated()


def test_postgres_settings_reject_a_non_identifier_table():
    with pytest.raises(ConfigError, match="PG_TABLE"):
        Settings(
            vector_store="postgres",
            database_url="postgresql://localhost/x",
            embedding_provider="openai",
            pg_table="chunks; DROP TABLE users",
        ).validated()
