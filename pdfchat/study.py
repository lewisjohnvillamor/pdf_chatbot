"""Study-tool generation: summaries, glossaries, flashcards and quizzes.

Every tool draws on retrieved passages rather than the raw document, for two
reasons: the material may exceed the context window, and grounding each item
in a specific excerpt lets the UI show the learner where an answer came from.

Model output is validated structurally — a malformed quiz question is dropped
rather than rendered as a broken widget.
"""

from __future__ import annotations

import logging

from .config import Settings
from .errors import ProviderError
from .llm import ChatModel, complete_json
from .models import Flashcard, GlossaryTerm, QuizQuestion, ScoredChunk, Usage
from .prompts import (
    FLASHCARD_PROMPT,
    GLOSSARY_PROMPT,
    QUIZ_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    build_sources_prefix,
)
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)

#: Probe queries used to gather a representative spread of a document when the
#: learner asks for study material without naming a topic.
_SURVEY_QUERIES = [
    "introduction overview purpose scope",
    "definition key term concept principle",
    "method process procedure steps how it works",
    "result finding conclusion implication",
    "example application case limitation exception",
]

MAX_ITEMS = 30


def _source_label(scored: list[ScoredChunk], marker: str) -> str:
    """Resolve an ``"S3"`` marker from a model response to a readable locator."""
    digits = str(marker).strip().lstrip("Ss")
    if digits.isdigit():
        index = int(digits) - 1
        if 0 <= index < len(scored):
            return scored[index].chunk.locator
    return ""


def gather_context(
    retriever: HybridRetriever,
    *,
    collection: str,
    doc_ids: list[str] | None = None,
    topic: str = "",
    limit: int = 12,
) -> tuple[list[ScoredChunk], Usage]:
    """Collect a broad, de-duplicated sample of passages to study from.

    A single query returns a narrow slice of a document. Running several probe
    queries and merging by chunk id gives coverage across the material, which
    is what a summary or quiz needs.
    """
    queries = [topic] if topic.strip() else _SURVEY_QUERIES
    merged: dict[str, ScoredChunk] = {}
    usage = Usage()
    for query in queries:
        result = retriever.retrieve(query, collection=collection, doc_ids=doc_ids)
        usage = usage.add(result.usage)
        for scored in result.chunks:
            existing = merged.get(scored.chunk.chunk_id)
            if existing is None or scored.score > existing.score:
                merged[scored.chunk.chunk_id] = scored
    # Present passages in document order so the summary reads front to back.
    ordered = sorted(merged.values(), key=lambda s: (s.chunk.filename, s.chunk.ordinal))
    return ordered[:limit], usage


def _generate(
    model: ChatModel,
    prompt: str,
    settings: Settings,
    sources: list[ScoredChunk],
    *,
    max_tokens: int | None = None,
) -> tuple[dict, Usage]:
    """Run a study-tool call with the passages as a cacheable prefix.

    The four tools run over the same gathered passages with different
    instructions, so after the first call the passages are cache reads.
    """
    return complete_json(
        model,
        SYSTEM_PROMPT,
        prompt,
        max_tokens=max_tokens or settings.max_output_tokens,
        cache_prefix=build_sources_prefix(sources),
    )


def make_summary(
    model: ChatModel, sources: list[ScoredChunk], settings: Settings, *, level: str = "Intermediate"
) -> tuple[str, Usage]:
    """Produce a cited study summary of the supplied passages."""
    if not sources:
        return "", Usage()
    prompt = SUMMARY_PROMPT.format(level=level.lower())
    completion = model.complete(
        SYSTEM_PROMPT,
        prompt,
        max_tokens=settings.max_output_tokens,
        cache_prefix=build_sources_prefix(sources),
    )
    return completion.text, completion.usage


def make_glossary(
    model: ChatModel, sources: list[ScoredChunk], settings: Settings, *, count: int = 12
) -> tuple[list[GlossaryTerm], Usage]:
    """Extract key terms with definitions drawn from the sources."""
    if not sources:
        return [], Usage()
    count = max(1, min(count, MAX_ITEMS))
    prompt = GLOSSARY_PROMPT.format(count=count)
    payload, usage = _generate(model, prompt, settings, sources)
    terms: list[GlossaryTerm] = []
    for item in payload.get("terms", []) or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if term and definition:
            terms.append(
                GlossaryTerm(
                    term=term,
                    definition=definition,
                    source=_source_label(sources, item.get("source", "")),
                )
            )
    logger.info("glossary_generated", extra={"terms": len(terms)})
    return terms[:count], usage


def make_flashcards(
    model: ChatModel,
    sources: list[ScoredChunk],
    settings: Settings,
    *,
    count: int = 10,
    level: str = "Intermediate",
) -> tuple[list[Flashcard], Usage]:
    """Generate question/answer flashcards grounded in the sources."""
    if not sources:
        return [], Usage()
    count = max(1, min(count, MAX_ITEMS))
    prompt = FLASHCARD_PROMPT.format(count=count, level=level.lower())
    payload, usage = _generate(model, prompt, settings, sources)
    cards: list[Flashcard] = []
    seen: set[str] = set()
    for item in payload.get("cards", []) or []:
        if not isinstance(item, dict):
            continue
        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        key = front.lower()
        if not front or not back or key in seen:
            continue
        seen.add(key)
        cards.append(
            Flashcard(front=front, back=back, source=_source_label(sources, item.get("source", "")))
        )
    logger.info("flashcards_generated", extra={"cards": len(cards)})
    return cards[:count], usage


def make_quiz(
    model: ChatModel,
    sources: list[ScoredChunk],
    settings: Settings,
    *,
    count: int = 5,
    level: str = "Intermediate",
) -> tuple[list[QuizQuestion], Usage]:
    """Generate validated multiple-choice questions with explanations."""
    if not sources:
        return [], Usage()
    count = max(1, min(count, MAX_ITEMS))
    prompt = QUIZ_PROMPT.format(count=count, level=level.lower())
    payload, usage = _generate(model, prompt, settings, sources)

    questions: list[QuizQuestion] = []
    for item in payload.get("questions", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question", "")).strip()
        options = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()]
        answer_index = item.get("answer_index")
        # A question with a duplicate option or an out-of-range answer is broken;
        # dropping it beats rendering a quiz the learner cannot pass.
        if not text or len(options) < 2 or len(set(options)) != len(options):
            continue
        if not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
            continue
        questions.append(
            QuizQuestion(
                question=text,
                options=options,
                answer_index=answer_index,
                explanation=str(item.get("explanation", "")).strip(),
                source=_source_label(sources, item.get("source", "")),
            )
        )
    dropped = len(payload.get("questions", []) or []) - len(questions)
    if dropped:
        logger.warning("quiz_items_dropped", extra={"dropped": dropped})
    if not questions:
        raise ProviderError(
            "The quiz could not be generated from this material. Try selecting a "
            "different document or narrowing the topic."
        )
    return questions[:count], usage
