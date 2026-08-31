"""Core dataclasses shared across ingestion, retrieval and generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Page:
    """One extracted PDF page after cleaning."""

    number: int  # 1-indexed, as printed in a PDF viewer
    text: str
    char_count: int = 0

    @property
    def has_text_layer(self) -> bool:
        return self.char_count > 0


@dataclass(slots=True)
class Document:
    """A single ingested PDF."""

    doc_id: str
    filename: str
    pages: list[Page]
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(page.char_count for page in self.pages)


@dataclass(slots=True)
class Chunk:
    """A retrievable passage with enough provenance to cite it."""

    chunk_id: str
    doc_id: str
    filename: str
    text: str
    page_start: int
    page_end: int
    ordinal: int
    section: str | None = None
    token_estimate: int = 0

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def locator(self) -> str:
        """Human-readable source reference, e.g. ``notes.pdf, pp. 3-4``."""
        base = f"{self.filename}, {self.page_label}"
        return f"{base} — {self.section}" if self.section else base

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ScoredChunk:
    """A chunk plus the retrieval scores that selected it."""

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rank: int = 0


@dataclass(frozen=True, slots=True)
class Citation:
    """A ``[S1]`` marker resolved back to the passage it points at."""

    marker: int
    chunk_id: str
    filename: str
    page_start: int
    page_end: int
    excerpt: str

    @property
    def label(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"[S{self.marker}] {self.filename}, {pages}"


@dataclass(slots=True)
class Usage:
    """Token counters and derived cost for one or more provider calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )


GroundingVerdict = Literal["grounded", "partially_grounded", "ungrounded", "unchecked"]


@dataclass(slots=True)
class Answer:
    """A generated answer with its evidence and grounding assessment."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[ScoredChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    verdict: GroundingVerdict = "unchecked"
    verdict_note: str = ""
    refused: bool = False
    follow_ups: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Flashcard:
    front: str
    back: str
    source: str = ""


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    source: str = ""

    @property
    def answer_text(self) -> str:
        if 0 <= self.answer_index < len(self.options):
            return self.options[self.answer_index]
        return ""


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    term: str
    definition: str
    source: str = ""
