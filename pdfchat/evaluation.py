"""Retrieval evaluation: measure the pipeline instead of asserting it works.

A RAG system without retrieval metrics is a system whose quality nobody knows.
Generation quality is downstream of retrieval — if the passage containing the
answer never reaches the prompt, no model and no prompt can recover it. So the
number that matters most is: *how often is the answer-bearing passage in the
top k?*

This module implements the standard ranking metrics and a runner that sweeps
configurations, so choices like ``HYBRID_DENSE_WEIGHT`` are set by measurement
rather than taste.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .models import ScoredChunk
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled question.

    Relevance is expressed as substrings that must appear in a retrieved
    passage rather than as chunk ids, so the gold set stays valid when chunking
    settings change — which is exactly when you most want to re-measure.
    """

    question: str
    must_contain: list[str]
    note: str = ""
    #: A question the corpus genuinely cannot answer. Retrieval always returns
    #: its nearest passages regardless, so refusing is the *generator's* job —
    #: these cases are excluded from ranking metrics and checked in the
    #: pipeline tests instead. Counting them as retrieval misses would
    #: understate recall by punishing the retriever for working correctly.
    negative: bool = False

    def is_relevant(self, chunk_text: str) -> bool:
        haystack = " ".join(chunk_text.lower().split())
        return any(needle.lower() in haystack for needle in self.must_contain)


def load_goldset(path: str | Path) -> list[EvalCase]:
    """Read a JSON gold set: ``[{"question": ..., "must_contain": [...]}, ...]``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=item["question"],
            must_contain=list(item["must_contain"]),
            note=item.get("note", ""),
            negative=bool(item.get("negative", False)),
        )
        for item in payload
    ]


# --- metrics ---------------------------------------------------------------
def recall_at_k(relevance: list[bool], k: int) -> float:
    """1.0 if any relevant passage is in the top k. The metric that matters most.

    Binary rather than proportional: for question answering, one passage
    carrying the answer is usually enough, so "did we find it at all" is the
    honest question.
    """
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    """1/rank of the first relevant passage; 0.0 if there is none."""
    for index, hit in enumerate(relevance, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int) -> float:
    """Normalised discounted cumulative gain over binary relevance.

    Unlike recall, this rewards ranking the relevant passage *higher*, which
    matters because earlier passages get more of the model's attention and
    survive context truncation.
    """
    gains = [1.0 / math.log2(i + 2) for i, hit in enumerate(relevance[:k]) if hit]
    if not gains:
        return 0.0
    ideal_count = min(sum(relevance), k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return sum(gains) / ideal if ideal else 0.0


@dataclass(slots=True)
class CaseResult:
    case: EvalCase
    relevance: list[bool]
    recall: float
    rr: float
    ndcg: float
    top_locator: str = ""

    @property
    def found(self) -> bool:
        return any(self.relevance)


@dataclass(slots=True)
class EvalResult:
    """Aggregate scores for one configuration."""

    label: str
    k: int
    cases: list[CaseResult] = field(default_factory=list)
    dense_weight: float | None = None
    skipped_negatives: int = 0
    #: Passages in the corpus. Recall is only meaningful when k is a small
    #: fraction of this — otherwise the retriever "finds" the answer merely by
    #: returning most of the corpus.
    corpus_size: int = 0

    def _mean(self, attribute: str) -> float:
        if not self.cases:
            return 0.0
        return sum(getattr(c, attribute) for c in self.cases) / len(self.cases)

    @property
    def recall(self) -> float:
        return self._mean("recall")

    @property
    def mrr(self) -> float:
        return self._mean("rr")

    @property
    def ndcg(self) -> float:
        return self._mean("ndcg")

    @property
    def misses(self) -> list[CaseResult]:
        """Cases where no relevant passage was retrieved — the actionable list."""
        return [case for case in self.cases if not case.found]

    @property
    def coverage(self) -> float:
        """Fraction of the corpus that top-k returns. High values invalidate recall."""
        return self.k / self.corpus_size if self.corpus_size else 0.0

    @property
    def coverage_is_confounding(self) -> bool:
        """True when k reaches so much of the corpus that recall is inflated."""
        return self.coverage > 0.25

    def row(self) -> str:
        flag = " !" if self.coverage_is_confounding else "  "
        coverage = f"{self.coverage * 100:>4.0f}%{flag}" if self.corpus_size else "    -  "
        return (
            f"{self.label:<26} {self.recall:>9.3f} {self.mrr:>7.3f} "
            f"{self.ndcg:>8.3f} {len(self.misses):>7} {coverage}"
        )


def score_case(case: EvalCase, chunks: list[ScoredChunk], k: int) -> CaseResult:
    relevance = [case.is_relevant(scored.chunk.text) for scored in chunks]
    return CaseResult(
        case=case,
        relevance=relevance,
        recall=recall_at_k(relevance, k),
        rr=reciprocal_rank(relevance),
        ndcg=ndcg_at_k(relevance, k),
        top_locator=chunks[0].chunk.locator if chunks else "",
    )


def evaluate(
    retriever: HybridRetriever,
    cases: list[EvalCase],
    *,
    collection: str,
    k: int,
    label: str = "config",
    dense_weight: float | None = None,
    corpus_size: int = 0,
) -> EvalResult:
    """Run every case through ``retriever`` and aggregate the scores."""
    scored = [case for case in cases if not case.negative]
    result = EvalResult(
        label=label,
        k=k,
        dense_weight=dense_weight,
        skipped_negatives=len(cases) - len(scored),
        corpus_size=corpus_size,
    )
    for case in scored:
        retrieved = retriever.retrieve(case.question, collection=collection)
        result.cases.append(score_case(case, retrieved.chunks, k))
    logger.info(
        "eval_complete",
        extra={"label": label, "recall": result.recall, "mrr": result.mrr, "cases": len(cases)},
    )
    return result


def sweep_dense_weight(
    store,
    embedder,
    settings: Settings,
    cases: list[EvalCase],
    *,
    collection: str,
    weights: list[float] | None = None,
) -> list[EvalResult]:
    """Measure retrieval across the keyword↔semantic blend.

    weight 0.0 = pure BM25, 1.0 = pure dense. The point is to find out where
    the corpus actually performs best rather than defaulting to 0.5 by habit.
    """
    weights = weights if weights is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    results: list[EvalResult] = []
    for weight in weights:
        tuned = settings.with_overrides(hybrid_dense_weight=weight)
        label = {0.0: "BM25 only", 1.0: "dense only"}.get(weight, f"hybrid w={weight:g}")
        results.append(
            evaluate(
                HybridRetriever(store, embedder, tuned),
                cases,
                collection=collection,
                k=tuned.top_k,
                label=label,
                dense_weight=weight,
                corpus_size=store.count(collection),
            )
        )
    return results


def configurations_are_indistinguishable(results: list[EvalResult]) -> bool:
    """True when every configuration produced identical scores.

    This is a finding, not a tie: it means the dense and lexical rankers are
    returning the same order, so fusion has nothing to fuse. Usually the
    embedder carries no signal beyond term overlap — the offline hashed
    embedder always behaves this way — and the sweep cannot answer the
    question it was asked.
    """
    if len(results) < 2:
        return False
    signature = {(round(r.recall, 6), round(r.mrr, 6), round(r.ndcg, 6)) for r in results}
    return len(signature) == 1


def format_table(results: list[EvalResult]) -> str:
    """Render a comparison table, marking the best configuration."""
    header = (
        f"{'configuration':<26} {'recall@k':>9} {'MRR':>7} {'nDCG@k':>8} {'misses':>7} {'cover':>7}"
    )
    lines = [header, "-" * len(header)]
    best = max(results, key=lambda r: (r.recall, r.mrr)) if results else None
    if configurations_are_indistinguishable(results):
        best = None  # nothing "won"; saying otherwise would be noise
    for result in results:
        marker = "  <- best" if result is best else ""
        lines.append(result.row() + marker)
    if any(r.coverage_is_confounding for r in results):
        lines.append("")
        lines.append(
            "  ! top-k reaches over 25% of the corpus in the flagged rows. Their recall\n"
            "    is inflated: a retriever scores well there by returning most of the\n"
            "    corpus, not by ranking well. Do not compare them against the others."
        )
    return "\n".join(lines)
