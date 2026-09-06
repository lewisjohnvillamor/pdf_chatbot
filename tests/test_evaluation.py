"""Tests for the evaluation harness.

Metrics that are silently wrong are worse than no metrics — they produce
confident, misleading tuning decisions. Every expected value here is computed
by hand from the metric's definition.
"""

from __future__ import annotations

import json
import math

import pytest

from pdfchat.evaluation import (
    CaseResult,
    EvalCase,
    EvalResult,
    configurations_are_indistinguishable,
    format_table,
    load_goldset,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_case,
)
from pdfchat.models import ScoredChunk
from tests.conftest import make_chunk


# --- metrics ---------------------------------------------------------------
def test_recall_is_binary_within_k():
    assert recall_at_k([True, False, False], 3) == 1.0
    assert recall_at_k([False, False, True], 3) == 1.0
    assert recall_at_k([False, False, False], 3) == 0.0


def test_recall_ignores_hits_beyond_k():
    # The hit is at rank 4; with k=3 it never reaches the prompt.
    assert recall_at_k([False, False, False, True], 3) == 0.0
    assert recall_at_k([False, False, False, True], 4) == 1.0


def test_reciprocal_rank_matches_its_definition():
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, True]) == 0.5
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_is_one_when_the_hit_is_first():
    assert ndcg_at_k([True, False, False], 3) == 1.0


def test_ndcg_discounts_lower_ranks():
    # Single hit at rank 2: gain 1/log2(3), ideal 1/log2(2) = 1.
    assert ndcg_at_k([False, True], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_one_when_all_hits_are_at_the_top():
    assert ndcg_at_k([True, True, False], 3) == pytest.approx(1.0)


def test_ndcg_is_zero_without_any_hit():
    assert ndcg_at_k([False, False], 2) == 0.0


def test_ndcg_ranks_earlier_hits_higher():
    assert ndcg_at_k([True, False, False], 3) > ndcg_at_k([False, False, True], 3)


# --- case scoring ----------------------------------------------------------
def test_relevance_matching_is_whitespace_and_case_insensitive():
    case = EvalCase("q", ["takes place in the MITOCHONDRIA"])
    assert case.is_relevant("...which\n  takes   place in the mitochondria and yields...")


def test_relevance_accepts_any_of_several_substrings():
    case = EvalCase("q", ["crossing over", "haploid gametes"])
    assert case.is_relevant("produces four haploid gametes")
    assert not case.is_relevant("mitosis produces diploid cells")


def test_score_case_records_the_top_locator():
    case = EvalCase("q", ["target text"])
    chunks = [
        ScoredChunk(chunk=make_chunk("d:00000", "irrelevant filler", page=1), score=1.0),
        ScoredChunk(chunk=make_chunk("d:00001", "contains target text here", page=2), score=0.9),
    ]
    result = score_case(case, chunks, k=2)
    assert result.relevance == [False, True]
    assert result.found
    assert result.rr == 0.5
    assert "p. 1" in result.top_locator


def test_score_case_with_no_results():
    result = score_case(EvalCase("q", ["anything"]), [], k=5)
    assert not result.found
    assert result.recall == 0.0 and result.rr == 0.0


# --- aggregation -----------------------------------------------------------
def _result(label: str, hits: list[list[bool]], corpus_size: int = 100) -> EvalResult:
    result = EvalResult(label=label, k=5, corpus_size=corpus_size)
    for relevance in hits:
        result.cases.append(
            CaseResult(
                case=EvalCase("q", ["x"]),
                relevance=relevance,
                recall=recall_at_k(relevance, 5),
                rr=reciprocal_rank(relevance),
                ndcg=ndcg_at_k(relevance, 5),
            )
        )
    return result


def test_aggregate_means_are_correct():
    result = _result("cfg", [[True], [False], [True], [True]])
    assert result.recall == 0.75
    assert len(result.misses) == 1


def test_empty_result_does_not_divide_by_zero():
    empty = EvalResult(label="cfg", k=5)
    assert empty.recall == 0.0 and empty.mrr == 0.0 and empty.ndcg == 0.0


def test_coverage_flags_a_confounded_configuration():
    confounded = _result("small corpus", [[True]], corpus_size=10)  # k=5 of 10
    honest = _result("large corpus", [[True]], corpus_size=100)
    assert confounded.coverage == 0.5
    assert confounded.coverage_is_confounding
    assert not honest.coverage_is_confounding


def test_format_table_warns_about_confounded_rows():
    table = format_table([_result("tiny", [[True]], corpus_size=8)])
    assert "!" in table
    assert "inflated" in table


def test_identical_configurations_are_detected():
    a = _result("a", [[True], [False]])
    b = _result("b", [[True], [False]])
    assert configurations_are_indistinguishable([a, b])


def test_differing_configurations_are_not_flagged():
    a = _result("a", [[True], [True]])
    b = _result("b", [[True], [False]])
    assert not configurations_are_indistinguishable([a, b])


def test_no_winner_is_declared_when_everything_ties():
    a, b = _result("a", [[True]]), _result("b", [[True]])
    assert "<- best" not in format_table([a, b])


# --- gold set --------------------------------------------------------------
def test_goldset_loads_and_marks_negatives(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            [
                {"question": "a", "must_contain": ["x"], "note": "n"},
                {"question": "b", "must_contain": [], "negative": True},
            ]
        )
    )
    cases = load_goldset(path)
    assert len(cases) == 2
    assert not cases[0].negative
    assert cases[1].negative


def test_shipped_goldset_is_well_formed():
    from pathlib import Path

    goldset = Path(__file__).resolve().parent.parent / "evals" / "goldset.json"
    cases = load_goldset(goldset)
    assert len(cases) >= 20, "the gold set must be big enough for the means to mean something"
    assert any(case.negative for case in cases), "a negative control is required"
    for case in cases:
        assert case.question.strip()
        # Every scored case needs something to match against.
        assert case.negative or case.must_contain


# --- the shared hashing embedder ------------------------------------------
def test_hashing_embedder_is_deterministic():
    """The eval harness depends on this: same corpus, same numbers, every run."""
    from evals.hashing import HashingEmbedder

    a, b = HashingEmbedder(128), HashingEmbedder(128)
    left, _ = a.embed_query("cellular respiration in the mitochondria")
    right, _ = b.embed_query("cellular respiration in the mitochondria")
    assert (left == right).all()


def test_hashing_embedder_vectors_are_unit_length():
    import numpy as np

    from evals.hashing import HashingEmbedder

    matrix, _ = HashingEmbedder(128).embed_documents(["alpha beta", "gamma delta epsilon"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_hashing_embedder_scores_shared_vocabulary_higher():
    from evals.hashing import HashingEmbedder

    e = HashingEmbedder(512)
    query, _ = e.embed_query("mitochondria produce ATP")
    docs, _ = e.embed_documents(
        ["the mitochondria produce ATP for the cell", "photosynthesis happens in chloroplasts"]
    )
    similarity = docs @ query[0]
    assert similarity[0] > similarity[1]


def test_hashing_embedder_handles_empty_input():
    from evals.hashing import HashingEmbedder

    matrix, usage = HashingEmbedder(64).embed_documents([])
    assert matrix.shape == (0, 64)
    assert usage.embedding_tokens == 0


def test_hashing_embedder_rejects_bad_dimensions():
    import pytest

    from evals.hashing import HashingEmbedder

    with pytest.raises(ValueError, match="dimensions must be positive"):
        HashingEmbedder(0)
