"""End-to-end: a real PDF goes in, a cited answer comes out.

Only the chat model is faked. Ingestion, cleaning, chunking, embedding,
storage, hybrid retrieval, citation resolution and grounding all run for real.
"""

from __future__ import annotations

import pytest

from pdfchat.indexing import build_index, corpus_fingerprint
from pdfchat.rag import RagPipeline
from pdfchat.retrieval import HybridRetriever
from pdfchat.stores.memory import MemoryVectorStore
from tests.conftest import FakeChatModel
from tests.test_ingest_export import FakeUpload, make_pdf

pytest.importorskip("reportlab", reason="reportlab builds the test PDFs")

BIOLOGY_PAGES = [
    [
        "Biology Revision Notes",
        "CHAPTER ONE: PHOTOSYNTHESIS",
        "Photosynthesis converts light energy into chemical energy.",
        "It occurs in the chloroplasts of plant cells and requires",
        "water and carbon dioxide as inputs.",
        "1",
    ],
    [
        "Biology Revision Notes",
        "CHAPTER TWO: RESPIRATION",
        "Cellular respiration releases energy stored in glucose.",
        "In eukaryotic cells it takes place in the mitochondria and",
        "produces about 30 molecules of ATP per glucose molecule.",
        "2",
    ],
    [
        "Biology Revision Notes",
        "CHAPTER THREE: ENZYMES",
        "Enzymes are biological catalysts that lower activation energy.",
        "Each enzyme is specific to a particular substrate shape.",
        "3",
    ],
]


@pytest.fixture()
def indexed(settings, fake_embedder):
    """A fully built index over a real generated PDF."""
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
    upload = FakeUpload("biology.pdf", make_pdf(BIOLOGY_PAGES))
    report = build_index(
        [upload],
        settings=settings,
        embedder=fake_embedder,
        store=store,
        collection="c",
    )
    return store, report


def test_index_report_describes_the_corpus(indexed):
    _store, report = indexed
    assert report.succeeded
    assert len(report.documents) == 1
    assert report.page_count == 3
    assert report.chunk_count > 0
    assert report.problems == []
    assert not report.from_cache


def test_running_headers_never_reach_the_index(indexed):
    store, _report = indexed
    chunks = store.all_chunks("c")
    assert chunks
    assert all("Biology Revision Notes" not in chunk.text for chunk in chunks)


def test_page_numbers_survive_as_metadata_not_as_text(indexed):
    store, _report = indexed
    pages = {chunk.page_start for chunk in store.all_chunks("c")}
    assert pages <= {1, 2, 3}
    assert pages, "every chunk must know which page it came from"


def test_question_produces_a_cited_answer(indexed, settings, fake_embedder):
    store, _report = indexed
    model = FakeChatModel(
        [
            "Cellular respiration happens in the mitochondria [S1].",
            '{"verdict":"grounded","unsupported_claims":[],"note":"Supported."}',
        ]
    )
    pipeline = RagPipeline(HybridRetriever(store, fake_embedder, settings), model, settings)
    answer = pipeline.answer("Where does cellular respiration take place?", collection="c")

    assert answer.citations, "the answer must resolve at least one source"
    assert answer.verdict == "grounded"
    assert not answer.refused
    cited = answer.citations[0]
    assert cited.filename == "biology.pdf"
    assert cited.page_start >= 1


def test_retrieved_passage_actually_contains_the_answer(indexed, settings, fake_embedder):
    store, _report = indexed
    retriever = HybridRetriever(store, fake_embedder, settings)
    result = retriever.retrieve("mitochondria ATP glucose", collection="c")
    assert result.chunks
    combined = " ".join(scored.chunk.text.lower() for scored in result.chunks)
    assert "mitochondria" in combined


def test_second_index_of_the_same_pdf_hits_the_cache(settings, fake_embedder, tmp_path):
    cached_settings = settings.with_overrides(cache_dir=tmp_path)
    upload = FakeUpload("biology.pdf", make_pdf(BIOLOGY_PAGES))

    first_store = MemoryVectorStore(dimensions=fake_embedder.dimensions, cache_dir=tmp_path)
    first = build_index(
        [upload],
        settings=cached_settings,
        embedder=fake_embedder,
        store=first_store,
        collection="c",
    )
    assert not first.from_cache

    second_store = MemoryVectorStore(dimensions=fake_embedder.dimensions, cache_dir=tmp_path)
    second = build_index(
        [upload],
        settings=cached_settings,
        embedder=fake_embedder,
        store=second_store,
        collection="c",
    )
    assert second.from_cache
    assert second.chunk_count == first.chunk_count
    assert second.usage.embedding_tokens == 0, "a cache hit must not re-embed"


def test_fingerprint_changes_when_chunking_settings_change(settings, fake_embedder):
    from pdfchat.ingest import load_pdf

    document = load_pdf(FakeUpload("biology.pdf", make_pdf(BIOLOGY_PAGES)), settings)
    base = corpus_fingerprint([document], settings, fake_embedder.name)
    resized = corpus_fingerprint(
        [document], settings.with_overrides(chunk_size=900), fake_embedder.name
    )
    remodelled = corpus_fingerprint([document], settings, "a-different-model")
    assert base != resized, "a chunk-size change must invalidate the cache"
    assert base != remodelled, "an embedding-model change must invalidate the cache"


def test_progress_callback_reaches_completion(settings, fake_embedder):
    updates: list[tuple[float, str]] = []
    build_index(
        [FakeUpload("biology.pdf", make_pdf(BIOLOGY_PAGES))],
        settings=settings,
        embedder=fake_embedder,
        store=MemoryVectorStore(dimensions=fake_embedder.dimensions),
        collection="c",
        progress=lambda fraction, message: updates.append((fraction, message)),
    )
    assert updates
    assert updates[-1][0] == 1.0


def test_unreadable_file_is_reported_not_raised(settings, fake_embedder):
    report = build_index(
        [FakeUpload("broken.pdf", b"not a pdf at all")],
        settings=settings,
        embedder=fake_embedder,
        store=MemoryVectorStore(dimensions=fake_embedder.dimensions),
        collection="c",
    )
    assert not report.succeeded
    assert report.problems
