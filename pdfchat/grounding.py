"""Post-generation grounding verification.

Citing a source is not the same as being supported by it. This module runs a
second, cheap model pass that compares the draft answer against the exact
excerpts it was given and reports whether each claim actually holds. It is the
difference between "the model said it cited page 12" and "the claim is on
page 12".

Deterministic checks run first and short-circuit the paid call: an answer with
no citations at all, or one that already declared the documents insufficient,
needs no verification.
"""

from __future__ import annotations

import logging

from .citations import has_any_citation
from .llm import ChatModel, extract_json
from .models import GroundingVerdict, ScoredChunk, Usage
from .prompts import GROUNDING_PROMPT, GROUNDING_SYSTEM, build_sources_prefix

logger = logging.getLogger(__name__)

REFUSAL_PREFIX = "the documents don't cover this"
_VALID_VERDICTS = {"grounded", "partially_grounded", "ungrounded"}
#: The check is a short structured judgement; it never needs a long budget.
_MAX_TOKENS = 700


def looks_like_refusal(answer: str) -> bool:
    """Did the model correctly decline for lack of source material?"""
    return answer.strip().lower().startswith(REFUSAL_PREFIX)


def verify(
    answer: str, sources: list[ScoredChunk], model: ChatModel
) -> tuple[GroundingVerdict, str, Usage]:
    """Judge whether ``answer`` is supported by ``sources``."""
    if not answer.strip() or not sources:
        return "unchecked", "", Usage()
    if looks_like_refusal(answer):
        return "grounded", "The assistant declined for lack of source material.", Usage()
    if not has_any_citation(answer):
        return (
            "ungrounded",
            "The answer cites no sources, so none of its claims can be traced back "
            "to your documents.",
            Usage(),
        )

    prompt = GROUNDING_PROMPT.format(answer=answer)
    try:
        completion = model.complete(
            GROUNDING_SYSTEM,
            prompt,
            max_tokens=_MAX_TOKENS,
            cache_prefix=build_sources_prefix(sources),
        )
        payload = extract_json(completion.text)
    except Exception:
        # A failed check must never block the answer the learner already has.
        logger.warning("grounding_check_failed", exc_info=True)
        return "unchecked", "The grounding check could not be completed.", Usage()

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        return (
            "unchecked",
            "The grounding check returned an unrecognised verdict.",
            completion.usage,
        )

    note = str(payload.get("note", "")).strip()
    unsupported = payload.get("unsupported_claims") or []
    if isinstance(unsupported, list) and unsupported:
        preview = "; ".join(str(claim) for claim in unsupported[:3])
        note = f"{note} Unsupported: {preview}" if note else f"Unsupported: {preview}"

    logger.info("grounding_verified", extra={"verdict": verdict})
    return verdict, note, completion.usage  # type: ignore[return-value]
