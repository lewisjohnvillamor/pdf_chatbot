from __future__ import annotations

from pdfchat.chunking import (
    MIN_CHUNK_CHARS,
    chunk_document,
    estimate_tokens,
    hamming,
    simhash,
)
from pdfchat.models import Document, Page


def test_chunks_carry_page_provenance(document, settings):
    chunks = chunk_document(document, settings)
    assert chunks
    for chunk in chunks:
        assert chunk.page_start >= 1
        assert chunk.page_end >= chunk.page_start
        assert chunk.filename == "biology.pdf"
        assert chunk.chunk_id.startswith("doc1:")


def test_chunk_ids_are_unique_and_ordinals_dense(document, settings):
    chunks = chunk_document(document, settings)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_section_heading_is_attached_to_following_text(document, settings):
    chunks = chunk_document(document, settings)
    sections = {c.section for c in chunks}
    assert "PHOTOSYNTHESIS" in sections


def test_heading_text_is_not_duplicated_into_the_body(document, settings):
    chunks = chunk_document(document, settings)
    first = chunks[0]
    assert first.section == "PHOTOSYNTHESIS"
    assert not first.text.startswith("PHOTOSYNTHESIS")


def test_respects_chunk_size_budget(document, settings):
    settings = settings.with_overrides(chunk_size=200, chunk_overlap=40)
    for chunk in chunk_document(document, settings):
        # Overlap is prepended, so allow one overlap window of slack.
        assert len(chunk.text) <= 200 + 40 + MIN_CHUNK_CHARS


def test_long_paragraph_is_split_on_sentence_boundaries(settings):
    body = " ".join(f"Sentence number {n} explains a distinct idea." for n in range(60))
    document = Document("d", "long.pdf", [Page(1, body, len(body))], "b" * 64, 100)
    chunks = chunk_document(document, settings.with_overrides(chunk_size=300, chunk_overlap=0))
    assert len(chunks) > 1
    # No chunk should begin mid-word.
    assert all(chunk.text[0].isupper() or chunk.text[0].isdigit() for chunk in chunks)


def test_near_duplicate_chunks_are_dropped(settings):
    boilerplate = (
        "This document is confidential and proprietary. Unauthorised distribution "
        "is strictly prohibited by the terms of the licence agreement in force."
    )
    pages = [Page(n, f"{boilerplate}\n\nUnique content for page {n}.", 200) for n in range(1, 6)]
    document = Document("d", "dup.pdf", pages, "c" * 64, 100)
    chunks = chunk_document(document, settings)
    repeats = sum(1 for chunk in chunks if "strictly prohibited" in chunk.text)
    assert repeats == 1


def test_empty_document_yields_no_chunks(settings):
    document = Document("d", "empty.pdf", [Page(1, "", 0)], "d" * 64, 0)
    assert chunk_document(document, settings) == []


def test_simhash_is_stable_and_discriminating():
    a = simhash("the mitochondria is the powerhouse of the cell")
    b = simhash("the mitochondria is the powerhouse of the cell")
    c = simhash("photosynthesis converts sunlight into chemical energy in plants")
    assert a == b
    assert hamming(a, c) > 3


def test_estimate_tokens_is_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100
