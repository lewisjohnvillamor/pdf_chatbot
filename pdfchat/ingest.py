"""PDF ingestion: read bytes, extract per-page text, clean, and validate.

Guards run before any expensive work so a 500 MB upload or an encrypted file
fails fast with an actionable message rather than mid-embedding.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import BinaryIO, Protocol

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .cleaning import clean_document_pages
from .config import Settings
from .errors import IngestionError
from .models import Document, Page

logger = logging.getLogger(__name__)


class UploadedFile(Protocol):
    """The subset of Streamlit's UploadedFile that ingestion depends on."""

    name: str

    def getvalue(self) -> bytes: ...


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_pages(stream: BinaryIO, filename: str, settings: Settings) -> tuple[list[str], dict]:
    try:
        reader = PdfReader(stream)
    except PdfReadError as exc:
        raise IngestionError(f"{filename} is not a readable PDF: {exc}") from exc

    if reader.is_encrypted:
        # Some PDFs are "encrypted" with an empty owner password and open fine.
        try:
            if reader.decrypt("") == 0:
                raise IngestionError(
                    f"{filename} is password-protected; remove the password first."
                )
        except (NotImplementedError, PdfReadError) as exc:
            raise IngestionError(f"{filename} uses an unsupported encryption scheme.") from exc

    page_count = len(reader.pages)
    if page_count == 0:
        raise IngestionError(f"{filename} contains no pages.")
    if page_count > settings.max_pages_per_doc:
        raise IngestionError(
            f"{filename} has {page_count} pages; the limit is {settings.max_pages_per_doc}. "
            "Split the file and upload the parts separately."
        )

    raw_pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception:  # pypdf raises assorted errors on malformed page trees
            logger.warning("page_extract_failed", extra={"file": filename, "page": index})
            raw_pages.append("")

    metadata: dict[str, str] = {}
    try:
        info = reader.metadata or {}
        for key in ("/Title", "/Author", "/Subject", "/Creator"):
            value = info.get(key)
            if value:
                metadata[key.lstrip("/").lower()] = str(value)
    except Exception:  # metadata is optional and frequently malformed
        logger.debug("metadata_read_failed", extra={"file": filename})
    return raw_pages, metadata


def load_pdf(file: UploadedFile, settings: Settings) -> Document:
    """Read one uploaded PDF into a cleaned :class:`Document`."""
    data = file.getvalue()
    size_bytes = len(data)
    if size_bytes == 0:
        raise IngestionError(f"{file.name} is empty.")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise IngestionError(
            f"{file.name} is {size_bytes / 1_048_576:.1f} MB; the limit is "
            f"{settings.max_upload_mb} MB."
        )

    digest = sha256_bytes(data)
    raw_pages, metadata = _read_pages(io.BytesIO(data), file.name, settings)
    cleaned = clean_document_pages(raw_pages)
    pages = [Page(number=p.number, text=p.text, char_count=len(p.text)) for p in cleaned]

    warnings: list[str] = []
    empty_pages = [
        p.number for p in pages if p.char_count < settings.min_chars_per_page_for_text_layer
    ]
    if len(empty_pages) == len(pages):
        raise IngestionError(
            f"{file.name} has no extractable text — it is most likely a scanned document. "
            "Run OCR on it (for example `ocrmypdf in.pdf out.pdf`) and upload the result."
        )
    if empty_pages:
        preview = ", ".join(str(n) for n in empty_pages[:10])
        suffix = "…" if len(empty_pages) > 10 else ""
        warnings.append(
            f"{len(empty_pages)} of {len(pages)} pages have little or no text "
            f"(pages {preview}{suffix}); they may be scans or images."
        )

    document = Document(
        doc_id=digest[:16],
        filename=file.name,
        pages=pages,
        sha256=digest,
        size_bytes=size_bytes,
        metadata=metadata,
        warnings=warnings,
    )
    logger.info(
        "pdf_ingested",
        extra={
            "file": file.name,
            "doc_id": document.doc_id,
            "pages": document.page_count,
            "chars": document.char_count,
            "size_bytes": size_bytes,
            "empty_pages": len(empty_pages),
        },
    )
    return document


def load_pdfs(files: list[UploadedFile], settings: Settings) -> tuple[list[Document], list[str]]:
    """Ingest several PDFs, collecting per-file errors instead of aborting.

    Duplicate uploads (same SHA-256) are skipped so re-processing does not
    double-weight a document in retrieval.
    """
    documents: list[Document] = []
    problems: list[str] = []
    seen: set[str] = set()
    for file in files:
        try:
            document = load_pdf(file, settings)
        except IngestionError as exc:
            problems.append(str(exc))
            continue
        if document.sha256 in seen:
            problems.append(f"{file.name} is a duplicate of an earlier upload; skipped.")
            continue
        seen.add(document.sha256)
        documents.append(document)
    return documents, problems
