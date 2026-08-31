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
