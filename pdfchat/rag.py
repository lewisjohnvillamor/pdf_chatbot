"""The question-answering pipeline: retrieve, generate, verify, cite.

This is the orchestration layer the UI calls. It keeps every step observable —
the caller receives the sources that were used, the token cost, and the
grounding verdict alongside the answer text, so nothing about how the answer
was produced is hidden from the learner.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .citations import extract_citations, find_invalid_markers, strip_invalid_markers
from .config import Settings
from .errors import ProviderError
from .grounding import verify
from .llm import ChatModel, extract_json
from .models import Answer, Usage
from .prompts import FOLLOW_UP_PROMPT, SYSTEM_PROMPT, build_answer_prompt, format_sources
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)

NO_SOURCES_MESSAGE = (
    "The documents don't cover this. Nothing in the uploaded material matched your "
    "question closely enough to answer it. Try rephrasing with terms the documents "
    "would actually use, or check that the right document is selected."
)


class RagPipeline:
    """Answers questions against an indexed corpus."""

    def __init__(
        self,
        retriever: HybridRetriever,
        model: ChatModel,
        settings: Settings,
    ):
        self._retriever = retriever
        self._model = model
        self._settings = settings

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        collection: str,
        doc_ids: list[str] | None = None,
        level: str = "Intermediate",
        history: list[tuple[str, str]] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Answer:
        """Answer ``question``, streaming tokens to ``on_token`` when provided."""
        question = question.strip()
        if not question:
            return Answer(question=question, text="Ask a question to get started.", refused=True)

        retrieval = self._retriever.retrieve(question, collection=collection, doc_ids=doc_ids)
        usage = retrieval.usage

        if retrieval.is_empty:
            logger.info("no_sources_retrieved", extra={"question_chars": len(question)})
            return Answer(
                question=question,
                text=NO_SOURCES_MESSAGE,
                usage=usage,
                refused=True,
                verdict="grounded",
                verdict_note="No source material matched the question.",
            )

        prompt = build_answer_prompt(question, retrieval.chunks, level=level, history=history)
        text = self._generate(prompt, on_token)
        usage = usage.add(self._model.last_usage())

        # A marker pointing at a source that was never supplied is a fabrication.
        invalid = find_invalid_markers(text, len(retrieval.chunks))
        if invalid:
            logger.warning("invalid_citation_markers", extra={"markers": invalid})
            text = strip_invalid_markers(text, len(retrieval.chunks))

        verdict, note, verify_usage = ("unchecked", "", Usage())
        if self._settings.enable_self_check:
            verdict, note, verify_usage = verify(text, retrieval.chunks, self._model)
            usage = usage.add(verify_usage)

        citations = extract_citations(text, retrieval.chunks)
        answer = Answer(
            question=question,
            text=text,
            citations=citations,
            sources=retrieval.chunks,
            usage=usage,
            verdict=verdict,
            verdict_note=note,
            refused=text.strip().lower().startswith("the documents don't cover this"),
        )
        logger.info(
            "answer_generated",
            extra={
                "verdict": verdict,
                "citations": len(citations),
                "sources": len(retrieval.chunks),
                "cost_usd": round(usage.cost_usd, 6),
                "output_tokens": usage.output_tokens,
            },
        )
        return answer

    def _generate(self, prompt: str, on_token: Callable[[str], None] | None) -> str:
        """Run generation, streaming when both the config and caller allow it."""
        if on_token is None or not self._settings.enable_streaming:
            return self._model.complete(SYSTEM_PROMPT, prompt).text
        pieces: list[str] = []
        for piece in self._model.stream(SYSTEM_PROMPT, prompt):
            pieces.append(piece)
            on_token(piece)
        return "".join(pieces)

    # ------------------------------------------------------------------
    def suggest_follow_ups(self, answer: Answer, *, limit: int = 3) -> list[str]:
        """Propose next questions that the same sources can answer."""
        if answer.refused or not answer.sources:
            return []
        prompt = FOLLOW_UP_PROMPT.format(
            sources=format_sources(answer.sources),
            question=answer.question,
            answer=answer.text[:2000],
        )
        try:
            completion = self._model.complete(SYSTEM_PROMPT, prompt, max_tokens=500)
            payload = extract_json(completion.text)
        except ProviderError:
            logger.info("follow_up_generation_failed", exc_info=True)
            return []
        questions = payload.get("questions") or []
        return [str(q).strip() for q in questions if str(q).strip()][:limit]
