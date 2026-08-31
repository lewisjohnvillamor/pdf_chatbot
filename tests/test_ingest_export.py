from __future__ import annotations

import io

import pytest

from pdfchat.errors import IngestionError
from pdfchat.export import (
    flashcards_to_tsv,
    glossary_to_markdown,
    quiz_to_markdown,
    transcript_to_markdown,
)
from pdfchat.ingest import load_pdf, load_pdfs
from pdfchat.models import Answer, Citation, Flashcard, GlossaryTerm, QuizQuestion

reportlab = pytest.importorskip("reportlab", reason="reportlab builds the test PDFs")


class FakeUpload:
    """Stands in for Streamlit's UploadedFile."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def make_pdf(pages: list[list[str]]) -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for lines in pages:
        y = 720
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture()
def sample_pdf() -> bytes:
    return make_pdf(
        [
            ["Acme Study Guide", "Chapter One", "Photosynthesis makes glucose.", "1"],
            ["Acme Study Guide", "Chapter Two", "Respiration releases energy.", "2"],
            ["Acme Study Guide", "Chapter Three", "Enzymes are catalysts.", "3"],
            ["Acme Study Guide", "Chapter Four", "DNA stores information.", "4"],
        ]
    )


def test_load_pdf_extracts_pages_and_metadata(sample_pdf, settings):
    document = load_pdf(FakeUpload("guide.pdf", sample_pdf), settings)
    assert document.page_count == 4
    assert document.filename == "guide.pdf"
    assert len(document.sha256) == 64
    assert "Photosynthesis makes glucose." in document.pages[0].text


def test_running_header_is_removed_during_ingestion(sample_pdf, settings):
    document = load_pdf(FakeUpload("guide.pdf", sample_pdf), settings)
    assert all("Acme Study Guide" not in page.text for page in document.pages)


def test_doc_id_is_derived_from_content(sample_pdf, settings):
    first = load_pdf(FakeUpload("a.pdf", sample_pdf), settings)
    second = load_pdf(FakeUpload("b.pdf", sample_pdf), settings)
    assert first.doc_id == second.doc_id, "identical bytes must produce one identity"


def test_empty_file_is_rejected(settings):
    with pytest.raises(IngestionError, match="is empty"):
        load_pdf(FakeUpload("empty.pdf", b""), settings)


def test_non_pdf_bytes_are_rejected(settings):
    with pytest.raises(IngestionError, match="not a readable PDF"):
        load_pdf(FakeUpload("fake.pdf", b"this is plain text, not a PDF"), settings)


def test_oversized_upload_is_rejected(sample_pdf, settings):
    tiny = settings.with_overrides(max_upload_mb=1)
    oversized = FakeUpload("big.pdf", sample_pdf + b"\x00" * (2 * 1024 * 1024))
    with pytest.raises(IngestionError, match="the limit is 1 MB"):
        load_pdf(oversized, tiny)


def test_page_limit_is_enforced(sample_pdf, settings):
    with pytest.raises(IngestionError, match="the limit is 2"):
        load_pdf(FakeUpload("guide.pdf", sample_pdf), settings.with_overrides(max_pages_per_doc=2))


def test_pdf_without_a_text_layer_gives_an_ocr_hint(settings):
    blank = make_pdf([[], [], []])
    with pytest.raises(IngestionError, match="ocrmypdf"):
        load_pdf(FakeUpload("scan.pdf", blank), settings)


def test_load_pdfs_collects_errors_without_aborting(sample_pdf, settings):
    files = [FakeUpload("good.pdf", sample_pdf), FakeUpload("bad.pdf", b"nope")]
    documents, problems = load_pdfs(files, settings)
    assert len(documents) == 1
    assert len(problems) == 1
    assert "bad.pdf" in problems[0]


def test_duplicate_uploads_are_skipped(sample_pdf, settings):
    files = [FakeUpload("a.pdf", sample_pdf), FakeUpload("copy.pdf", sample_pdf)]
    documents, problems = load_pdfs(files, settings)
    assert len(documents) == 1
    assert any("duplicate" in problem for problem in problems)


# --- exports ---------------------------------------------------------------
def test_flashcard_tsv_is_anki_shaped():
    cards = [Flashcard("Front?", "Back\nwith newline", "guide.pdf, p. 2")]
    row = flashcards_to_tsv(cards).strip().split("\t")
    assert row[0] == "Front?"
    assert "<br>" in row[1], "raw newlines would break Anki's record separator"
    assert row[2] == "guide.pdf, p. 2"


def test_glossary_markdown_includes_terms_and_sources():
    output = glossary_to_markdown([GlossaryTerm("ATP", "Energy currency.", "guide.pdf, p. 3")])
    assert "**ATP**" in output
    assert "guide.pdf, p. 3" in output


def test_quiz_markdown_hides_answers_when_asked():
    question = QuizQuestion("Q?", ["a", "b", "c", "d"], 1, "Because b.", "guide.pdf, p. 1")
    with_key = quiz_to_markdown([question])
    without_key = quiz_to_markdown([question], include_answers=False)
    assert "Answer key" in with_key and "Because b." in with_key
    assert "Answer key" not in without_key


def test_transcript_includes_citations_and_verdict():
    answer = Answer(
        question="What is ATP?",
        text="ATP is the energy currency [S1].",
        citations=[Citation(1, "doc1:0", "guide.pdf", 3, 3, "excerpt")],
        verdict="grounded",
        verdict_note="All claims supported.",
    )
    output = transcript_to_markdown([answer])
    assert "## Q: What is ATP?" in output
    assert "[S1] guide.pdf, p. 3" in output
    assert "grounded" in output
