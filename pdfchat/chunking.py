"""Structure-aware chunking with page provenance and near-duplicate collapsing.

The tutorial-grade approach — split a concatenated blob every N characters —
destroys two things retrieval depends on: it cuts sentences in half, and it
loses the page number needed to cite the answer. This module instead walks the
document page by page, tracks the enclosing section heading, packs whole
paragraphs into chunks, and records the page span each chunk covers.

Boilerplate that survives cleaning (repeated disclaimers, copyright blocks) is
collapsed with a SimHash-based near-duplicate check so the same passage is not
embedded, retrieved and paid for many times over.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from .cleaning import looks_like_heading
from .config import Settings
from .models import Chunk, Document

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_WORD = re.compile(r"[a-z0-9]+")

#: Chunks shorter than this are merged into a neighbour rather than kept alone.
MIN_CHUNK_CHARS = 120
#: Hamming distance below which two 64-bit SimHashes count as near-duplicates.
SIMHASH_THRESHOLD = 3
#: Only paragraphs at least this long are considered for duplicate removal, so
#: short legitimate repeats ("None.", "See above.") are never dropped.
MIN_DEDUP_CHARS = 60


def estimate_tokens(text: str) -> int:
    """Cheap provider-agnostic token estimate (~4 characters per token).

    Used for budgeting and UI display only; billing always uses the token
    counts the provider reports back.
    """
    return max(1, len(text) // 4)


def simhash(text: str, *, bits: int = 64) -> int:
    """64-bit SimHash over word tokens, for near-duplicate detection."""
    vector = [0] * bits
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return 0
    for token in tokens:
        digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        for bit in range(bits):
            vector[bit] += 1 if digest >> bit & 1 else -1
    result = 0
    for bit in range(bits):
        if vector[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(slots=True)
class _Block:
    """A paragraph-sized unit of text with the page and section it came from."""

    text: str
    page: int
    section: str | None


def _split_blocks(document: Document) -> list[_Block]:
    """Break a document into paragraph blocks, carrying section headings forward."""
    blocks: list[_Block] = []
    section: str | None = None
    for page in document.pages:
        if not page.text.strip():
            continue
        for paragraph in page.text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines = paragraph.split("\n")
            # A short leading line that reads like a heading becomes the section
            # label for everything that follows.
            if looks_like_heading(lines[0]):
                section = lines[0].strip()
                body = "\n".join(lines[1:]).strip()
                if not body:
                    continue
                paragraph = body
            blocks.append(_Block(text=paragraph, page=page.number, section=section))
    return blocks


def _drop_duplicate_blocks(blocks: list[_Block]) -> list[_Block]:
    """Drop paragraphs that repeat across pages, keeping the first occurrence.

    Per-page disclaimers and copyright notices survive header/footer stripping
    because they sit in the body. Removing them here — before packing — stops
    them from being embedded once per page and from crowding out real content
    in retrieval.
    """
    kept: list[_Block] = []
    fingerprints: list[int] = []
    for block in blocks:
        if len(block.text) < MIN_DEDUP_CHARS:
            kept.append(block)
            continue
        fingerprint = simhash(block.text)
        if any(hamming(fingerprint, seen) <= SIMHASH_THRESHOLD for seen in fingerprints):
            continue
        fingerprints.append(fingerprint)
        kept.append(block)
    return kept


def _split_oversized(block: _Block, limit: int) -> list[_Block]:
    """Split a paragraph longer than ``limit`` on sentence boundaries."""
    if len(block.text) <= limit:
        return [block]
    pieces: list[_Block] = []
    current = ""
    for sentence in _SENTENCE_END.split(block.text):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(_Block(current.strip(), block.page, block.section))
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    # A single sentence longer than the limit (tables, long URLs) is hard-cut.
    if len(current) > limit:
        for start in range(0, len(current), limit):
            pieces.append(_Block(current[start : start + limit].strip(), block.page, block.section))
    elif current.strip():
        pieces.append(_Block(current.strip(), block.page, block.section))
    return [p for p in pieces if p.text]


def _overlap_tail(text: str, overlap: int) -> str:
    """Take the last ``overlap`` characters, starting at a sentence boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    match = _SENTENCE_END.search(tail)
    return tail[match.end() :] if match else tail


def chunk_document(document: Document, settings: Settings) -> list[Chunk]:
    """Turn one document into overlapping, page-attributed chunks."""
    blocks: list[_Block] = []
    for block in _drop_duplicate_blocks(_split_blocks(document)):
        blocks.extend(_split_oversized(block, settings.chunk_size))

    chunks: list[Chunk] = []
    buffer: list[_Block] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        text = "\n\n".join(b.text for b in buffer).strip()
        if not text:
            buffer, buffer_len = [], 0
            return
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{ordinal:05d}",
                doc_id=document.doc_id,
                filename=document.filename,
                text=text,
                page_start=min(b.page for b in buffer),
                page_end=max(b.page for b in buffer),
                ordinal=ordinal,
                section=buffer[0].section,
                token_estimate=estimate_tokens(text),
            )
        )
        # Seed the next chunk with the tail of this one so a fact split across
        # the boundary is still retrievable from either side.
        tail = _overlap_tail(text, settings.chunk_overlap)
        last = buffer[-1]
        buffer = [_Block(tail, last.page, last.section)] if tail else []
        buffer_len = len(tail)

    for block in blocks:
        if buffer and buffer_len + len(block.text) > settings.chunk_size:
            flush()
        buffer.append(block)
        buffer_len += len(block.text) + 2
    flush()

    merged = _merge_tiny_chunks(chunks)
    deduped = _drop_near_duplicates(merged)
    logger.info(
        "document_chunked",
        extra={
            "doc_id": document.doc_id,
            "file": document.filename,
            "chunks": len(deduped),
            "dropped_duplicates": len(merged) - len(deduped),
        },
    )
    return deduped


def _merge_tiny_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Fold sub-threshold chunks into the previous one from the same document."""
    if not chunks:
        return []
    out: list[Chunk] = []
    for chunk in chunks:
        if out and len(chunk.text) < MIN_CHUNK_CHARS and out[-1].doc_id == chunk.doc_id:
            previous = out[-1]
            previous.text = f"{previous.text}\n\n{chunk.text}"
            previous.page_end = max(previous.page_end, chunk.page_end)
            previous.token_estimate = estimate_tokens(previous.text)
            continue
        out.append(chunk)
    return out


def _drop_near_duplicates(chunks: list[Chunk]) -> list[Chunk]:
    """Remove chunks that are near-identical to one already kept."""
    kept: list[Chunk] = []
    hashes: list[int] = []
    for chunk in chunks:
        fingerprint = simhash(chunk.text)
        if any(hamming(fingerprint, seen) <= SIMHASH_THRESHOLD for seen in hashes):
            continue
        hashes.append(fingerprint)
        kept.append(chunk)
    # Ordinals must stay dense after removals so chunk ids remain stable.
    for position, chunk in enumerate(kept):
        chunk.ordinal = position
    return kept


def chunk_documents(documents: list[Document], settings: Settings) -> list[Chunk]:
    """Chunk a whole corpus."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, settings))
    return chunks
