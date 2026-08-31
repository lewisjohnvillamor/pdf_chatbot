"""Text normalization and structural repair for PDF-extracted text.

PDF extraction produces text that is hostile to retrieval: words split across
line breaks by hyphenation, ligatures encoded as single glyphs, running
headers and footers repeated on every page, page numbers interleaved with
prose, soft hyphens, and hard-wrapped lines that break sentences mid-clause.
Embedding that raw output wastes tokens and pollutes similarity scores.

This module fixes those problems before anything is chunked or embedded. Every
transformation is a pure function over strings so each one is independently
testable, and :func:`clean_document_pages` composes them in the order that
matters (cross-page analysis first, per-page repair second).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Ligatures and typographic glyphs that PDF fonts emit as single code points.
_GLYPH_REPLACEMENTS = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": " - ",
    "−": "-",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": "",
    "‌": "",
    "‍": "",
    "﻿": "",
    "­": "",
    "…": "...",
    "•": "- ",
    "●": "- ",
    "▪": "- ",
    "·": "- ",
}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
# Page-number-only lines, with or without decorations ("- 12 -", "Page 12 of 40").
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:[-–—\[\(]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?"
    r"\s*(?:[-–—\]\)])?\s*$",
    re.IGNORECASE,
)
_ROMAN_NUMERAL_LINE = re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE)
# Repeated dot/underscore leaders from tables of contents.
_LEADER_DOTS = re.compile(r"[.․‧_]{4,}")
_URL_SPLIT = re.compile(r"(https?://\S+?)\s+(?=\S)")

_HEADING_MAX_WORDS = 14
_NUMBERED_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.|[IVXLC]+\.)\s+\S")


def _apply_glyph_map(text: str) -> str:
    return text.translate(str.maketrans(_GLYPH_REPLACEMENTS))


def normalize_unicode(text: str) -> str:
    """Fold compatibility glyphs, ligatures and exotic whitespace to plain ASCII-ish text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _apply_glyph_map(text)
    text = _CONTROL_CHARS.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by end-of-line hyphenation."""
    # Python's `re` has no \p{Ll}; fall back to an ASCII-class pattern.
    return re.sub(r"(\w)-\n([a-z])", r"\1\2", text)


def unwrap_soft_linebreaks(text: str) -> str:
    """Join hard-wrapped lines that continue the same sentence.

    Blank lines, list markers and lines starting with a capital are treated as
    genuine breaks so paragraph and list structure survives.
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not out:
            out.append(stripped)
            continue
        previous = out[-1]
        # A line continues the previous one when the previous line simply ran
        # out of width: no terminal punctuation, and neither side is structural.
        continues = (
            previous
            and stripped
            and not previous.endswith((".", "!", "?", ":", ";"))
            and not _is_list_item(stripped)
            and not looks_like_heading(stripped)
            and not looks_like_heading(previous)
        )
        if continues:
            out[-1] = f"{previous} {stripped}"
        else:
            out.append(stripped)
    return "\n".join(out)


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*•]|\(?[a-z0-9]{1,3}[.)])\s+\S", line, re.IGNORECASE))


def strip_page_furniture(text: str) -> str:
    """Drop standalone page numbers, roman-numeral folios and TOC leader dots."""
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        if len(stripped) <= 7 and _ROMAN_NUMERAL_LINE.match(stripped):
            continue
        kept.append(_LEADER_DOTS.sub(" ", line))
    return "\n".join(kept)


def tidy_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines, and repair punctuation spacing."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _URL_SPLIT.sub(r"\1", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _edge_indices(lines: list[str], edge_lines: int) -> list[int]:
    """Positions of the non-empty lines near the top and bottom of a page.

    On a page with too few lines to separate edge from body, only the very
    first and last lines count — otherwise every line looks like furniture and
    genuine content gets stripped.
    """
    filled = [i for i, line in enumerate(lines) if line.strip()]
    if not filled:
        return []
    span = edge_lines if len(filled) > 2 * edge_lines else 1
    return sorted(set(filled[:span]) | set(filled[-span:]))


def find_repeated_lines(
    pages: list[str], *, min_pages: int = 3, ratio: float = 0.6, edge_lines: int = 3
) -> set[str]:
    """Identify running headers/footers by their repetition across pages.

    Only the first and last ``edge_lines`` of each page are considered, so a
    sentence that legitimately recurs in body text is never stripped. A line
    must appear near the edge of at least ``ratio`` of the pages (and at least
    ``min_pages`` of them) to be treated as furniture.
    """
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.split("\n")
        edges = [lines[i].strip() for i in _edge_indices(lines, edge_lines)]
        # Digits vary page to page ("Chapter 3 | 47"); normalize them out.
        counts.update({re.sub(r"\d+", "#", line) for line in edges if 3 <= len(line) <= 120})
    threshold = max(min_pages, int(len(pages) * ratio))
    return {line for line, count in counts.items() if count >= threshold}


def remove_repeated_lines(text: str, repeated: set[str], *, edge_lines: int = 3) -> str:
    """Strip previously identified running headers/footers from one page."""
    if not repeated:
        return text
    lines = text.split("\n")
    keep_flags = [True] * len(lines)
    for i in _edge_indices(lines, edge_lines):
        if re.sub(r"\d+", "#", lines[i].strip()) in repeated:
            keep_flags[i] = False
    return "\n".join(line for line, keep in zip(lines, keep_flags, strict=True) if keep)


def looks_like_heading(line: str) -> bool:
    """Heuristic: is this line a section heading rather than prose?"""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    words = stripped.split()
    if len(words) > _HEADING_MAX_WORDS:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    if _NUMBERED_HEADING.match(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    if all(c.isupper() for c in letters) and len(letters) >= 3:
        return True
    if len(words) == 1:
        # A lone capitalised word on its own line ("Introduction") is a heading.
        return len(stripped) >= 3 and stripped[0].isupper()
    # Title Case with no sentence-ending punctuation.
    capitalized = sum(1 for w in words if w[:1].isupper())
    return len(words) >= 2 and capitalized / len(words) >= 0.75


@dataclass(frozen=True, slots=True)
class CleanedPage:
    number: int
    text: str
    dropped_chars: int


def clean_page(text: str, repeated: set[str] | None = None) -> str:
    """Run the full per-page repair pipeline."""
    text = normalize_unicode(text)
    text = remove_repeated_lines(text, repeated or set())
    text = strip_page_furniture(text)
    text = dehyphenate(text)
    text = unwrap_soft_linebreaks(text)
    return tidy_whitespace(text)


def clean_document_pages(raw_pages: list[str]) -> list[CleanedPage]:
    """Clean every page of a document, using cross-page analysis for furniture.

    Header/footer detection needs the whole document, so it runs first over
    lightly normalized text; the per-page pipeline then uses its result.
    """
    normalized = [normalize_unicode(page or "") for page in raw_pages]
    repeated = find_repeated_lines(normalized)
    cleaned: list[CleanedPage] = []
    for index, page in enumerate(normalized, start=1):
        result = clean_page(page, repeated)
        cleaned.append(
            CleanedPage(number=index, text=result, dropped_chars=max(0, len(page) - len(result)))
        )
    return cleaned
