"""Resolve and validate the ``[S1]`` markers a model writes into an answer.

Two failure modes matter. A *hallucinated marker* points at a source that was
never supplied — that is a hard error and the marker is stripped. An
*uncited answer* contains factual claims with no marker at all, which the
grounding check treats as suspicious. Both are detected here rather than left
for the reader to notice.
"""

from __future__ import annotations

import re

from .models import Citation, ScoredChunk

#: Matches [S1], [S2, S4] and [S1][S3] alike.
MARKER = re.compile(r"\[S(\d+(?:\s*,\s*S?\d+)*)\]")
_EXCERPT_CHARS = 400


def _marker_numbers(text: str) -> list[int]:
    """Every source number referenced in ``text``, in order of first appearance."""
    seen: list[int] = []
    for match in MARKER.finditer(text):
        for piece in match.group(1).split(","):
            digits = piece.strip().lstrip("Ss")
            if digits.isdigit():
                number = int(digits)
                if number not in seen:
                    seen.append(number)
    return seen


def extract_citations(answer: str, sources: list[ScoredChunk]) -> list[Citation]:
    """Resolve markers in ``answer`` against the sources that were supplied."""
    citations: list[Citation] = []
    for number in _marker_numbers(answer):
        if not 1 <= number <= len(sources):
            continue  # hallucinated marker; strip_invalid_markers removes it
        chunk = sources[number - 1].chunk
        excerpt = chunk.text[:_EXCERPT_CHARS]
        if len(chunk.text) > _EXCERPT_CHARS:
            excerpt += "…"
        citations.append(
            Citation(
                marker=number,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                excerpt=excerpt,
            )
        )
    return citations


def find_invalid_markers(answer: str, source_count: int) -> list[int]:
    """Marker numbers that point outside the supplied source list."""
    return [n for n in _marker_numbers(answer) if not 1 <= n <= source_count]


def strip_invalid_markers(answer: str, source_count: int) -> str:
    """Remove markers that reference sources which were never provided."""

    def replace(match: re.Match[str]) -> str:
        kept = [
            piece.strip()
            for piece in match.group(1).split(",")
            if piece.strip().lstrip("Ss").isdigit()
            and 1 <= int(piece.strip().lstrip("Ss")) <= source_count
        ]
        if not kept:
            return ""
        normalized = ", ".join(p if p.upper().startswith("S") else f"S{p}" for p in kept)
        return f"[{normalized}]"

    cleaned = MARKER.sub(replace, answer)
    # Collapse the double spaces a removed marker leaves behind.
    return re.sub(r" {2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",")


def has_any_citation(answer: str) -> bool:
    return bool(MARKER.search(answer))
