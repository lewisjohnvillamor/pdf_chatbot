from __future__ import annotations

import numpy as np
import pytest

from pdfchat.config import Settings
from pdfchat.history import (
    MemoryHistoryStore,
    Turn,
    build_history_store,
    new_conversation_id,
    turn_from_answer,
)
from pdfchat.llm import Completion, extract_json
from pdfchat.models import Answer, Citation, Usage
from pdfchat.service import build_service
from pdfchat.stores.postgres import _from_pgvector, _to_pgvector


# --- history ---------------------------------------------------------------
def test_memory_history_roundtrip():
    store = MemoryHistoryStore()
    conversation = new_conversation_id()
    store.append(conversation, "c", Turn("user", "What is ATP?"))
    store.append(conversation, "c", Turn("assistant", "Energy currency [S1]."))
    turns = store.load(conversation)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].content == "Energy currency [S1]."


def test_history_is_isolated_per_conversation():
    store = MemoryHistoryStore()
    store.append("one", "c", Turn("user", "first"))
    store.append("two", "c", Turn("user", "second"))
    assert len(store.load("one")) == 1
    assert store.load("unknown") == []


def test_clear_removes_only_that_conversation():
    store = MemoryHistoryStore()
    store.append("one", "c", Turn("user", "hi"))
    store.append("two", "c", Turn("user", "hi"))
    store.clear("one")
    assert store.load("one") == []
    assert len(store.load("two")) == 1


def test_clearing_a_conversation_removes_its_transcript():
    """The sidebar's "Clear conversation" calls this; it must actually delete."""
    store = MemoryHistoryStore()
    store.append("c1", "col", Turn("user", "a question"))
    store.append("c1", "col", Turn("assistant", "an answer [S1]."))
    assert len(store.load("c1")) == 2
    store.clear("c1")
    assert store.load("c1") == []


def test_conversation_ids_are_unique():
    assert len({new_conversation_id() for _ in range(50)}) == 50


def test_turn_from_answer_carries_citations_and_cost():
    answer = Answer(
        question="Q",
        text="A [S1].",
        citations=[Citation(1, "doc1:0", "notes.pdf", 4, 5, "excerpt")],
        usage=Usage(cost_usd=0.002),
        verdict="grounded",
    )
    turn = turn_from_answer(answer)
    assert turn.role == "assistant"
    assert turn.verdict == "grounded"
    assert turn.cost_usd == 0.002
    assert turn.citations[0]["page_start"] == 4
    assert turn.citations[0]["filename"] == "notes.pdf"


def test_history_store_falls_back_to_memory_without_a_database():
    store = build_history_store(Settings(persist_conversations=True, database_url=None))
    assert isinstance(store, MemoryHistoryStore)


def test_history_store_is_memory_when_persistence_is_off():
    settings = Settings(persist_conversations=False, database_url="postgresql://localhost/x")
    assert isinstance(build_history_store(settings), MemoryHistoryStore)


# --- service wiring --------------------------------------------------------
def test_build_service_wires_the_full_graph(monkeypatch):
    from tests.conftest import FakeChatModel

    monkeypatch.setattr("pdfchat.service.build_chat_model", lambda _s: FakeChatModel())
    settings = Settings(
        chat_provider="anthropic",
        anthropic_api_key="k",
        embedding_provider="none",
        vector_store="memory",
    ).validated()

    service = build_service(settings)
    assert service.chunk_count() == 0
    assert service.documents() == []
    assert service.pipeline is not None
    assert service.retriever is not None
    service.reset()
    service.close()


def test_service_reset_clears_the_corpus(monkeypatch, fake_embedder):
    from pdfchat.stores.base import ChunkRecord
    from tests.conftest import FakeChatModel, make_chunk

    monkeypatch.setattr("pdfchat.service.build_chat_model", lambda _s: FakeChatModel())
    monkeypatch.setattr("pdfchat.service.build_embedder", lambda _s: fake_embedder)
    settings = Settings(anthropic_api_key="k", embedding_provider="none").validated()
    service = build_service(settings)

    vectors, _ = fake_embedder.embed_documents(["some text"])
    service.store.add(
        [ChunkRecord(chunk=make_chunk("doc1:00000", "some text"), embedding=vectors[0])],
        collection=service.collection,
    )
    assert service.chunk_count() == 1
    service.reset()
    assert service.chunk_count() == 0


# --- pgvector serialization ------------------------------------------------
def test_pgvector_literal_roundtrip():
    original = np.array([0.5, -0.25, 0.125], dtype=np.float32)
    literal = _to_pgvector(original)
    assert literal.startswith("[") and literal.endswith("]")
    assert np.allclose(_from_pgvector(literal), original)


def test_pgvector_literal_handles_a_realistic_dimension():
    original = np.random.default_rng(0).normal(size=1536).astype(np.float32)
    restored = _from_pgvector(_to_pgvector(original))
    assert restored.shape == (1536,)
    assert np.allclose(restored, original, atol=1e-5)


# --- usage / cost accounting ----------------------------------------------
def test_usage_addition_accumulates_every_counter():
    total = Usage(input_tokens=10, output_tokens=5, cost_usd=0.1, calls=1).add(
        Usage(input_tokens=3, output_tokens=2, cache_read_tokens=7, cost_usd=0.05, calls=1)
    )
    assert (total.input_tokens, total.output_tokens, total.calls) == (13, 7, 2)
    assert total.cache_read_tokens == 7
    assert total.cost_usd == pytest.approx(0.15)


def test_completion_carries_usage():
    completion = Completion(text="hello", usage=Usage(output_tokens=2), stop_reason="end_turn")
    assert completion.usage.output_tokens == 2


def test_extract_json_recovers_from_preamble_and_fences():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here it is: {"b": [1, 2]}') == {"b": [1, 2]}
    assert extract_json('{"c": {"nested": true}}') == {"c": {"nested": True}}


def test_extract_json_rejects_unparseable_output():
    from pdfchat.errors import ProviderError

    with pytest.raises(ProviderError):
        extract_json("no json here at all")
