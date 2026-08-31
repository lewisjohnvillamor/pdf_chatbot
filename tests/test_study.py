from __future__ import annotations

import json

import pytest

from pdfchat.errors import ProviderError
from pdfchat.models import ScoredChunk
from pdfchat.study import make_flashcards, make_glossary, make_quiz, make_summary
from tests.conftest import FakeChatModel, make_chunk


@pytest.fixture()
def sources() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=make_chunk("doc1:00000", "Mitochondria produce ATP.", page=3), score=1.0),
        ScoredChunk(
            chunk=make_chunk("doc1:00001", "Chloroplasts capture light.", page=5), score=0.9
        ),
    ]


def test_summary_is_returned_verbatim(sources, settings):
    model = FakeChatModel(["## Summary\n- Point one [S1]"])
    text, usage = make_summary(model, sources, settings)
    assert "Point one [S1]" in text
    assert usage.calls == 1


def test_summary_of_no_sources_is_empty(settings):
    assert make_summary(FakeChatModel(), [], settings)[0] == ""


def test_glossary_resolves_source_markers_to_locators(sources, settings):
    payload = {"terms": [{"term": "ATP", "definition": "Energy currency.", "source": "S1"}]}
    terms, _ = make_glossary(FakeChatModel([json.dumps(payload)]), sources, settings)
    assert terms[0].term == "ATP"
    assert terms[0].source == "biology.pdf, p. 3"


def test_glossary_drops_incomplete_entries(sources, settings):
    payload = {
        "terms": [
            {"term": "ATP", "definition": ""},
            {"term": "", "definition": "x"},
            {"term": "NADPH", "definition": "A carrier.", "source": "S2"},
        ]
    }
    terms, _ = make_glossary(FakeChatModel([json.dumps(payload)]), sources, settings)
    assert [t.term for t in terms] == ["NADPH"]


def test_flashcards_are_deduplicated(sources, settings):
    payload = {
        "cards": [
            {"front": "What is ATP?", "back": "Energy currency.", "source": "S1"},
            {"front": "what is atp?", "back": "Duplicate.", "source": "S1"},
            {"front": "What do chloroplasts do?", "back": "Capture light.", "source": "S2"},
        ]
    }
    cards, _ = make_flashcards(FakeChatModel([json.dumps(payload)]), sources, settings)
    assert len(cards) == 2


def test_flashcard_count_is_capped(sources, settings):
    payload = {"cards": [{"front": f"Q{i}", "back": f"A{i}"} for i in range(20)]}
    cards, _ = make_flashcards(FakeChatModel([json.dumps(payload)]), sources, settings, count=5)
    assert len(cards) == 5


def test_quiz_questions_are_validated(sources, settings):
    payload = {
        "questions": [
            {
                "question": "Valid?",
                "options": ["a", "b", "c", "d"],
                "answer_index": 2,
                "explanation": "Because.",
                "source": "S1",
            },
            {
                "question": "Out of range?",
                "options": ["a", "b"],
                "answer_index": 7,
                "explanation": "",
            },
            {
                "question": "Duplicate options?",
                "options": ["a", "a"],
                "answer_index": 0,
                "explanation": "",
            },
            {
                "question": "Too few options?",
                "options": ["only"],
                "answer_index": 0,
                "explanation": "",
            },
        ]
    }
    questions, _ = make_quiz(FakeChatModel([json.dumps(payload)]), sources, settings)
    assert len(questions) == 1
    assert questions[0].answer_text == "c"
    assert questions[0].source == "biology.pdf, p. 3"


def test_quiz_raises_when_nothing_survives_validation(sources, settings):
    payload = {"questions": [{"question": "Bad", "options": ["a"], "answer_index": 5}]}
    with pytest.raises(ProviderError, match="could not be generated"):
        make_quiz(FakeChatModel([json.dumps(payload)]), sources, settings)


def test_invalid_json_raises_a_provider_error(sources, settings):
    with pytest.raises(ProviderError, match="did not return valid JSON"):
        make_glossary(FakeChatModel(["totally not json"]), sources, settings)


def test_study_tools_return_empty_without_sources(settings):
    assert make_glossary(FakeChatModel(), [], settings)[0] == []
    assert make_flashcards(FakeChatModel(), [], settings)[0] == []
    assert make_quiz(FakeChatModel(), [], settings)[0] == []
