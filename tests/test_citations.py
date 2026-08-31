from __future__ import annotations

from pdfchat.citations import (
    extract_citations,
    find_invalid_markers,
    has_any_citation,
    strip_invalid_markers,
)
from pdfchat.models import ScoredChunk
from tests.conftest import make_chunk


def sources(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=make_chunk(f"doc1:{i:05d}", f"Body text {i}", page=i + 1), score=1.0)
        for i in range(n)
    ]


def test_extract_citations_resolves_markers_to_pages():
    citations = extract_citations("Claim one [S1]. Claim two [S2].", sources(3))
    assert [c.marker for c in citations] == [1, 2]
    assert citations[0].page_start == 1
    assert citations[1].page_start == 2


def test_extract_citations_handles_grouped_markers():
    citations = extract_citations("Both sources agree [S1, S3].", sources(3))
    assert [c.marker for c in citations] == [1, 3]


def test_extract_citations_deduplicates_repeated_markers():
    citations = extract_citations("First [S1]. Again [S1]. Still [S1].", sources(2))
    assert [c.marker for c in citations] == [1]


def test_extract_citations_ignores_out_of_range_markers():
    assert extract_citations("Fabricated [S9].", sources(2)) == []


def test_find_invalid_markers_reports_fabrications():
    assert find_invalid_markers("Real [S1]. Fake [S8].", 2) == [8]


def test_strip_invalid_markers_removes_only_the_bad_ones():
    cleaned = strip_invalid_markers("Real [S1]. Fake [S8]. Mixed [S2, S7].", 2)
    assert "[S1]" in cleaned
    assert "[S8]" not in cleaned
    assert "[S7]" not in cleaned
    assert "[S2]" in cleaned


def test_strip_invalid_markers_tidies_orphaned_punctuation():
    assert strip_invalid_markers("A claim [S9].", 1) == "A claim."


def test_citation_label_formats_page_ranges():
    citation = extract_citations("Text [S1].", sources(1))[0]
    assert citation.label == "[S1] biology.pdf, p. 1"


def test_has_any_citation():
    assert has_any_citation("Grounded [S1].")
    assert not has_any_citation("Ungrounded claim with no marker.")
