"""Export study material to formats learners already use.

Flashcards go out as TSV, which Anki imports directly. Quizzes and transcripts
go out as Markdown so they can be printed, committed, or pasted into notes.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime

from .models import Answer, Flashcard, GlossaryTerm, QuizQuestion


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def flashcards_to_tsv(cards: list[Flashcard]) -> str:
    """Anki-importable TSV: front, back, source. Tabs, no header."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    for card in cards:
        # Anki treats newlines as record separators; HTML breaks render correctly.
        writer.writerow(
            [
                card.front.replace("\n", "<br>"),
                card.back.replace("\n", "<br>"),
                card.source,
            ]
        )
    return buffer.getvalue()


def flashcards_to_markdown(cards: list[Flashcard]) -> str:
    lines = [f"# Flashcards\n\n_Generated {_timestamp()}_\n"]
    for index, card in enumerate(cards, start=1):
        lines.append(f"### {index}. {card.front}\n")
        lines.append(f"{card.back}\n")
        if card.source:
            lines.append(f"> Source: {card.source}\n")
    return "\n".join(lines)


def glossary_to_markdown(terms: list[GlossaryTerm]) -> str:
    lines = [f"# Glossary\n\n_Generated {_timestamp()}_\n"]
    for term in terms:
        lines.append(f"**{term.term}** — {term.definition}")
        if term.source:
            lines.append(f"  \n_Source: {term.source}_")
        lines.append("")
    return "\n".join(lines)


def quiz_to_markdown(questions: list[QuizQuestion], *, include_answers: bool = True) -> str:
    lines = [f"# Quiz\n\n_Generated {_timestamp()}_\n"]
    for index, question in enumerate(questions, start=1):
        lines.append(f"**{index}. {question.question}**\n")
        for option_index, option in enumerate(question.options):
            lines.append(f"   {chr(65 + option_index)}. {option}")
        lines.append("")
    if include_answers:
        lines.append("\n---\n\n## Answer key\n")
        for index, question in enumerate(questions, start=1):
            letter = chr(65 + question.answer_index)
            lines.append(f"**{index}. {letter}** — {question.answer_text}")
            if question.explanation:
                lines.append(f"  \n{question.explanation}")
            if question.source:
                lines.append(f"  \n_Source: {question.source}_")
            lines.append("")
    return "\n".join(lines)


def transcript_to_markdown(answers: list[Answer], *, title: str = "Study session") -> str:
    """Render a full Q&A transcript with citations and grounding verdicts."""
    lines = [f"# {title}\n\n_Exported {_timestamp()}_\n"]
    for answer in answers:
        lines.append(f"## Q: {answer.question}\n")
        lines.append(f"{answer.text}\n")
        if answer.citations:
            lines.append("**Sources**\n")
            seen: set[int] = set()
            for citation in answer.citations:
                if citation.marker in seen:
                    continue
                seen.add(citation.marker)
                lines.append(f"- {citation.label}")
            lines.append("")
        if answer.verdict != "unchecked":
            note = f" — {answer.verdict_note}" if answer.verdict_note else ""
            lines.append(f"_Grounding check: {answer.verdict.replace('_', ' ')}{note}_\n")
        lines.append("---\n")
    return "\n".join(lines)
