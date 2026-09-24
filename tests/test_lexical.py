from __future__ import annotations

from pdfchat.lexical import BM25, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cell IS a Powerhouse") == ["cell", "powerhouse"]


def test_tokenize_keeps_hyphenated_and_numeric_terms():
    assert "co2" in tokenize("plants absorb CO2")
    assert "well-known" in tokenize("a well-known result")


def test_bm25_ranks_the_relevant_document_first():
    corpus = [
        tokenize("mitochondria produce ATP through oxidative phosphorylation"),
        tokenize("photosynthesis converts light into glucose inside chloroplasts"),
        tokenize("the cell membrane regulates transport of ions"),
    ]
    index = BM25(corpus)
    assert index.top_n("ATP mitochondria", 3)[0][0] == 0
    assert index.top_n("chloroplasts glucose", 3)[0][0] == 1


def test_bm25_returns_nothing_for_unmatched_query():
    index = BM25([tokenize("alpha beta gamma")])
    assert index.top_n("completely unrelated terminology", 5) == []


def test_bm25_handles_empty_corpus():
    index = BM25([])
    assert len(index) == 0
    assert index.top_n("anything", 5) == []


def test_bm25_favours_rare_terms_over_common_ones():
    corpus = [tokenize("common word appears everywhere here")] * 5
    corpus.append(tokenize("common word plus rareterminology"))
    index = BM25(corpus)
    assert index.top_n("rareterminology", 1)[0][0] == 5


# --- structured identifiers ------------------------------------------------
def test_clause_identifiers_survive_tokenization():
    """The plain word tokenizer reduced 7(a), 7(b) and 7(c) all to "article"."""
    assert "7(c)" in tokenize("What does Article 7(c) allow?")
    assert "12(a)" in tokenize("filed under Article 12(a) within five days")


def test_sibling_clauses_are_distinguishable():
    a, b, c = (tokenize(f"Article 7({x}) applies here") for x in "abc")
    assert a != b != c
    assert len({tuple(a), tuple(b), tuple(c)}) == 3, "sibling clauses collided"


def test_identifiers_are_emitted_alongside_words_not_instead_of_them():
    tokens = tokenize("Article 7(c) permits an extension")
    assert "7(c)" in tokens
    assert "article" in tokens, "the ordinary words must still match"


def test_identifier_spacing_is_normalised():
    assert tokenize("Article 7 (c)")[0] == tokenize("Article 7(c)")[0]


def test_dotted_and_section_references_are_kept():
    assert "3.1" in tokenize("see section 3.1 of the handbook")
    assert "12.4.2" in tokenize("clause 12.4.2 applies")


def test_ordinary_text_is_unaffected():
    assert tokenize("The Cell IS a Powerhouse") == ["cell", "powerhouse"]


def test_bm25_can_now_separate_sibling_clauses():
    corpus = [
        tokenize("Article 7(a) requires submissions in electronic form."),
        tokenize("Article 7(b) states late filing incurs a penalty of 250 euros."),
        tokenize("Article 7(c) permits an extension of up to 30 days."),
    ]
    index = BM25(corpus)
    assert index.top_n("What does Article 7(c) allow?", 3)[0][0] == 2
    assert index.top_n("What is the penalty under Article 7(b)?", 3)[0][0] == 1
