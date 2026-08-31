"""Prompt templates.

Every prompt here enforces the same contract: answer only from the supplied
sources, cite them with ``[S1]``-style markers, and say so plainly when the
sources do not contain the answer. Grounding is a prompt property first and a
verification property second — :mod:`pdfchat.grounding` checks the result.
"""

from __future__ import annotations

from .models import ScoredChunk

#: How much detail the answer should carry, chosen by the learner.
EXPLAIN_LEVELS: dict[str, str] = {
    "Beginner": (
        "Explain as if to a curious beginner with no background in the subject. "
        "Define every technical term the first time it appears, use short sentences "
        "and a concrete everyday analogy where one genuinely fits."
    ),
    "Intermediate": (
        "Explain to a student who knows the basics of the field. Use correct "
        "terminology and briefly define anything specialised."
    ),
    "Expert": (
        "Explain to a domain expert. Be precise and concise, use standard "
        "terminology without defining it, and preserve exact figures and caveats."
    ),
}

SYSTEM_PROMPT = """You are a study assistant that answers strictly from a set of \
excerpts taken from the learner's own PDF documents.

Rules you must follow without exception:

1. Ground every factual claim in the provided sources. Never use outside knowledge \
to state a fact about the documents' subject matter, even if you are confident it \
is true.
2. Cite with bracketed markers that match the source labels, like [S1] or [S2, S4]. \
Put the marker immediately after the sentence it supports. Every paragraph that \
states a fact must carry at least one marker.
3. If the sources do not contain the answer, say exactly what is missing and stop. \
Begin that response with "The documents don't cover this." Do not guess, and do not \
pad the answer with general knowledge.
4. If the sources conflict, present both readings and cite each one.
5. Quote exact figures, dates, names and formulas rather than paraphrasing them.
6. Never invent a source marker. Only use markers that appear in the context below.

Format answers in Markdown. Lead with the direct answer in one or two sentences, \
then add supporting detail. Keep it focused: a good answer is as short as the \
question allows."""


def format_sources(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as labelled, citable source blocks."""
    blocks: list[str] = []
    for index, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        header = f"[S{index}] {chunk.filename} — {chunk.page_label}"
        if chunk.section:
            header += f" — section: {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_answer_prompt(
    question: str,
    chunks: list[ScoredChunk],
    *,
    level: str = "Intermediate",
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Assemble the user-turn prompt: sources, then history, then the question."""
    parts = [
        "Here are the excerpts from the learner's documents.\n",
        "<sources>",
        format_sources(chunks) or "(no sources retrieved)",
        "</sources>\n",
    ]
    if history:
        recent = "\n".join(f"Learner: {q}\nAssistant: {a}" for q, a in history[-3:])
        parts.append(f"<recent_conversation>\n{recent}\n</recent_conversation>\n")
    parts.append(EXPLAIN_LEVELS.get(level, EXPLAIN_LEVELS["Intermediate"]))
    parts.append(f"\n<question>\n{question}\n</question>")
    parts.append(
        "\nAnswer the question using only the sources above, citing them with [S1]-style markers."
    )
    return "\n".join(parts)


GROUNDING_SYSTEM = """You are a strict fact-checker. You compare a draft answer \
against the source excerpts it claims to be based on, and you report only what the \
sources actually support. You never use outside knowledge."""

GROUNDING_PROMPT = """<sources>
{sources}
</sources>

<draft_answer>
{answer}
</draft_answer>

Check every factual claim in the draft answer against the sources.

Reply with a single JSON object and nothing else:
{{"verdict": "grounded" | "partially_grounded" | "ungrounded",
  "unsupported_claims": ["..."],
  "note": "one sentence explaining the verdict"}}

Use "grounded" when every claim is supported by the sources. Use \
"partially_grounded" when the main answer holds but some detail is not supported. \
Use "ungrounded" when the central claim is unsupported. Ignore hedging language, \
restated questions and formatting; judge only factual content."""


SUMMARY_PROMPT = """<sources>
{sources}
</sources>

Write a study summary of the material above for a {level} learner.

Structure it as:
- **In one sentence** — what this material is about.
- **Key points** — 4 to 8 bullets, each citing its source with [S1]-style markers.
- **Worth remembering** — 2 to 4 facts, definitions or formulas most likely to be \
tested or reused.

Use only the sources. Cite every bullet."""


GLOSSARY_PROMPT = """<sources>
{sources}
</sources>

Extract the key terms a learner must understand to follow this material.

Reply with a single JSON object and nothing else:
{{"terms": [{{"term": "...", "definition": "...", "source": "S1"}}]}}

Rules: between 5 and {count} terms; define each in one or two sentences using only \
the sources; set "source" to the marker of the excerpt the definition comes from; \
prefer terms the material itself defines or relies on heavily. If the material \
defines fewer than 5 terms, return only the ones it genuinely defines."""


FLASHCARD_PROMPT = """<sources>
{sources}
</sources>

Write {count} flashcards that test understanding of this material for a {level} \
learner.

Reply with a single JSON object and nothing else:
{{"cards": [{{"front": "...", "back": "...", "source": "S1"}}]}}

Rules: the front is one clear question or prompt; the back is a complete but \
concise answer drawn only from the sources; set "source" to the marker of the \
supporting excerpt. Test understanding and application, not trivia — avoid \
questions answerable by matching a single word. Do not duplicate cards."""


QUIZ_PROMPT = """<sources>
{sources}
</sources>

Write {count} multiple-choice questions on this material for a {level} learner.

Reply with a single JSON object and nothing else:
{{"questions": [{{"question": "...", "options": ["A", "B", "C", "D"],
  "answer_index": 0, "explanation": "...", "source": "S1"}}]}}

Rules: exactly four options each; exactly one correct; "answer_index" is the \
0-based index of the correct option. Wrong options must be plausible and drawn from \
the same material — never absurd or obviously padded. The explanation says why the \
correct option is right and cites its excerpt. Vary which index is correct across \
questions. Every question must be answerable from the sources alone."""


FOLLOW_UP_PROMPT = """<sources>
{sources}
</sources>

<question>{question}</question>
<answer>{answer}</answer>

Suggest 3 follow-up questions the learner could usefully ask next about this \
material. Each must be answerable from the sources above.

Reply with a single JSON object and nothing else:
{{"questions": ["...", "...", "..."]}}"""
