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


# --- banded near-duplicate lookup ------------------------------------------
def test_banding_is_lossless_not_approximate():
    """Banding must reject exactly what a full pairwise scan would reject.

    This is the whole justification for the optimisation: it is a bucketing of
    the same comparison, not a cheaper approximation of it. If it ever starts
    missing pairs, boilerplate silently returns to the index.
    """
    import random

    from pdfchat.chunking import SIMHASH_THRESHOLD, _NearDuplicateFilter

    random.seed(7)
    vocab = [f"term{i}" for i in range(500)]
    texts = [" ".join(random.choice(vocab) for _ in range(20)) for _ in range(400)]
    texts += texts[::37]  # inject genuine duplicates
    random.shuffle(texts)
    fingerprints = [simhash(t) for t in texts]

    pairwise: list[int] = []
    for fp in fingerprints:
        if not any(hamming(fp, kept) <= SIMHASH_THRESHOLD for kept in pairwise):
            pairwise.append(fp)

    banded: list[int] = []
    seen = _NearDuplicateFilter()
    for fp in fingerprints:
        if seen.is_duplicate(fp):
            continue
        seen.add(fp)
        banded.append(fp)

    assert banded == pairwise


def test_near_duplicates_always_share_a_band():
    """The pigeonhole property the losslessness rests on."""
    import random

    from pdfchat.chunking import SIMHASH_BANDS, SIMHASH_THRESHOLD, _band_keys

    assert SIMHASH_BANDS > SIMHASH_THRESHOLD
    random.seed(11)
    base = simhash("cellular respiration releases energy from glucose molecules")
    for _ in range(200):
        flipped = base
        for bit in random.sample(range(64), SIMHASH_THRESHOLD):
            flipped ^= 1 << bit
        assert hamming(base, flipped) <= SIMHASH_THRESHOLD
        assert set(_band_keys(base)) & set(_band_keys(flipped)), "a close pair shared no band"


def test_filter_accepts_a_genuinely_different_fingerprint():
    from pdfchat.chunking import _NearDuplicateFilter

    seen = _NearDuplicateFilter()
    a = simhash("photosynthesis happens in the chloroplasts of plant cells")
    b = simhash("the appeals committee reviews sanctions on a quarterly basis")
    seen.add(a)
    assert not seen.is_duplicate(b)
    assert seen.is_duplicate(a)


def test_boilerplate_removal_still_works_end_to_end(settings):
    """The behaviour the optimisation must not change."""
    boilerplate = (
        "This document is confidential and proprietary. Unauthorised distribution "
        "is strictly prohibited by the terms of the licence agreement in force."
    )
    pages = [Page(n, f"{boilerplate}\n\nUnique content for page {n}.", 200) for n in range(1, 8)]
    document = Document("d", "dup.pdf", pages, "c" * 64, 100)
    chunks = chunk_document(document, settings)
    assert sum(1 for c in chunks if "strictly prohibited" in c.text) == 1
