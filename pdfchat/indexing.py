"""Corpus building: ingest → clean → chunk → embed → store.

Embedding is the expensive step, so a corpus is keyed by a *fingerprint* over
its document hashes and the settings that affect vectors. Re-uploading the same
PDFs with the same configuration is then a cache hit and costs nothing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from .chunking import chunk_documents
from .config import Settings
from .embeddings import Embedder
from .ingest import UploadedFile, load_pdfs
from .logging_setup import log_duration
from .models import Document, Usage
from .stores.base import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)

#: Embed in batches so progress is reportable and memory stays bounded.
EMBED_BATCH = 128


def corpus_fingerprint(documents: list[Document], settings: Settings, embedder_name: str) -> str:
    """Stable id for "these documents, chunked and embedded this way".

    Any change that would alter the stored vectors — a different document, a
    different chunk size, a different embedding model — must produce a
    different fingerprint, or a stale cache would be served.
    """
    parts = sorted(document.sha256 for document in documents)
    parts.extend(
        [
            embedder_name,
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            # Cleaning and chunking logic version — bump when their behaviour changes.
            "pipeline-v2",
        ]
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


@dataclass(slots=True)
class IndexReport:
    """What happened during one indexing run."""

    documents: list[Document] = field(default_factory=list)
    chunk_count: int = 0
    usage: Usage = field(default_factory=Usage)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    from_cache: bool = False
    fingerprint: str = ""

    @property
    def page_count(self) -> int:
        return sum(document.page_count for document in self.documents)

    @property
    def succeeded(self) -> bool:
        return self.chunk_count > 0


def build_index(
    files: list[UploadedFile],
    *,
    settings: Settings,
    embedder: Embedder,
    store: VectorStore,
    collection: str,
    progress: ProgressFn | None = None,
) -> IndexReport:
    """Ingest ``files`` into ``store`` under ``collection``."""
    report = IndexReport()
    notify = progress or (lambda _fraction, _message: None)

    notify(0.05, "Reading PDFs…")
    with log_duration(logger, "ingest_batch", files=len(files)) as fields:
        documents, problems = load_pdfs(files, settings)
        fields["documents"] = len(documents)
    report.documents = documents
    report.problems = problems
    for document in documents:
        report.warnings.extend(document.warnings)
    if not documents:
        return report

    fingerprint = corpus_fingerprint(documents, settings, embedder.name)
    report.fingerprint = fingerprint

    # A cached in-memory index restores instantly and skips all embedding cost.
    loader = getattr(store, "load", None)
    if callable(loader) and loader(collection, fingerprint):
        report.chunk_count = store.count(collection)
        report.from_cache = True
        notify(1.0, f"Restored {report.chunk_count} passages from cache.")
        return report

    notify(0.25, "Cleaning and splitting text…")
    with log_duration(logger, "chunk_batch") as fields:
        chunks = chunk_documents(documents, settings)
        fields["chunks"] = len(chunks)
    if not chunks:
        report.problems.append("No text could be extracted from these documents after cleaning.")
        return report

    records: list[ChunkRecord] = []
    if embedder.dimensions > 0:
        total = len(chunks)
        for start in range(0, total, EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors, usage = embedder.embed_documents([chunk.text for chunk in batch])
            report.usage = report.usage.add(usage)
            records.extend(
                ChunkRecord(chunk=chunk, embedding=vectors[i]) for i, chunk in enumerate(batch)
            )
            done = min(start + EMBED_BATCH, total)
            notify(0.3 + 0.6 * done / total, f"Embedding passages… {done}/{total}")
    else:
        records = [ChunkRecord(chunk=chunk) for chunk in chunks]
        notify(0.9, "Building keyword index (embeddings disabled)…")

    notify(0.95, "Saving index…")
    store.add(records, collection=collection)
    report.chunk_count = store.count(collection)

    saver = getattr(store, "save", None)
    if callable(saver):
        saver(collection, fingerprint)

    notify(1.0, f"Indexed {report.chunk_count} passages from {len(documents)} document(s).")
    logger.info(
        "index_built",
        extra={
            "documents": len(documents),
            "chunks": report.chunk_count,
            "cost_usd": round(report.usage.cost_usd, 6),
            "fingerprint": fingerprint,
        },
    )
    return report


# Declared after use for readability; Python resolves annotations lazily here.
from collections.abc import Callable  # noqa: E402

ProgressFn = Callable[[float, str], None]
