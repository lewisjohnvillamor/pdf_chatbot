"""Cross-encoder reranking: the precision stage after recall.

Retrieval is a recall problem — cast a wide net with BM25 and embeddings and
accept some noise. Reranking is a precision problem: take those candidates and
order them properly.

The difference is architectural, not incidental. A bi-encoder (what embeddings
are) encodes the query and the passage *separately* and compares two vectors,
so it can never model how a specific query term relates to a specific passage
term. A cross-encoder feeds query and passage through the model *together* and
attends across both, which is what lets it tell "Article 7(b)" from
"Article 7(c)" — a distinction two independently-computed vectors cannot
represent.

That accuracy costs a forward pass per candidate, so it is only affordable on
the shortlist retrieval already produced. The pipeline is therefore:

    BM25 + dense  ->  RRF fusion  ->  RERANK  ->  MMR  ->  top-k

Three backends are provided: a local cross-encoder (no API cost, needs
sentence-transformers), an LLM-based scorer (works with any configured
provider, no extra dependency), and a no-op passthrough.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from .config import Settings
from .errors import ConfigError
from .models import ScoredChunk, Usage

logger = logging.getLogger(__name__)

#: Passage text beyond this is truncated before scoring. Cross-encoders have a
#: fixed input window (typically 512 tokens) and silently truncate anyway; doing
#: it explicitly keeps the cost predictable.
MAX_PASSAGE_CHARS = 2000


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved candidates by relevance to the query."""

    name: str

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> tuple[list[ScoredChunk], Usage]:
        """Return the ``top_k`` most relevant candidates, best first."""


class NullReranker:
    """Passthrough. Keeps fusion order — the pipeline default when disabled."""

    name = "none"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> tuple[list[ScoredChunk], Usage]:
        return candidates[:top_k], Usage()


class CrossEncoderReranker:
    """Local cross-encoder scoring via sentence-transformers.

    Runs on CPU in tens of milliseconds for a few dozen candidates, costs
    nothing per query, and keeps document text on your own hardware — which
    matters when the whole point of self-hosting is that the material does not
    leave the building.
    """

    def __init__(self, settings: Settings):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigError(
                "RERANKER=cross-encoder requires sentence-transformers. Install it "
                "with: pip install -r requirements-local.txt"
            ) from exc
        self.name = settings.reranker_model
        # max_length caps the joined query+passage pair, not the passage alone.
        self._model = CrossEncoder(self.name, max_length=512)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> tuple[list[ScoredChunk], Usage]:
        if not candidates:
            return [], Usage()
        pairs = [(query, scored.chunk.text[:MAX_PASSAGE_CHARS]) for scored in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ordered = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
        )
        results: list[ScoredChunk] = []
        for rank, (scored, score) in enumerate(ordered[:top_k], start=1):
            # Replace the fusion score with the reranker's, keeping the
            # component scores so the UI can still explain the retrieval.
            results.append(
                ScoredChunk(
                    chunk=scored.chunk,
                    score=float(score),
                    dense_score=scored.dense_score,
                    lexical_score=scored.lexical_score,
                    rank=rank,
                )
            )
        logger.info("reranked", extra={"backend": "cross-encoder", "candidates": len(candidates)})
        return results, Usage()


class LLMReranker:
    """Scores passages with the configured chat model.

    Needs no extra dependency and works with either provider, but costs a call
    per query and adds latency. Prefer the local cross-encoder when you can
    install it; this exists for deployments that cannot.
    """

    name = "llm"

    _SYSTEM = (
        "You rate how well a passage answers a question. You output only JSON. You never explain."
    )

    def __init__(self, model):
        self._model = model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> tuple[list[ScoredChunk], Usage]:
        if not candidates:
            return [], Usage()

        from .llm import extract_json

        listing = "\n\n".join(
            f"[{index}] {scored.chunk.text[:MAX_PASSAGE_CHARS]}"
            for index, scored in enumerate(candidates)
        )
        prompt = (
            f"<question>{query}</question>\n\n<passages>\n{listing}\n</passages>\n\n"
            "Rate how well each passage answers the question, from 0 (irrelevant) "
            "to 10 (contains the answer directly).\n\n"
            'Reply with one JSON object and nothing else: {"scores": {"0": 7, "1": 0, ...}}\n'
            "Include every passage index. Judge only whether the passage answers "
            "the question — not whether it is well written."
        )
        try:
            completion = self._model.complete(self._SYSTEM, prompt, max_tokens=1000)
            payload = extract_json(completion.text)
            raw = payload.get("scores") or {}
            scores = {int(key): float(value) for key, value in raw.items()}
        except Exception:
            # A failed rerank must degrade to fusion order, never to no results.
            logger.warning("llm_rerank_failed_falling_back", exc_info=True)
            return candidates[:top_k], Usage()

        ordered = sorted(
            enumerate(candidates), key=lambda pair: scores.get(pair[0], -1.0), reverse=True
        )
        results = [
            ScoredChunk(
                chunk=scored.chunk,
                score=scores.get(index, 0.0),
                dense_score=scored.dense_score,
                lexical_score=scored.lexical_score,
                rank=rank,
            )
            for rank, (index, scored) in enumerate(ordered[:top_k], start=1)
        ]
        logger.info("reranked", extra={"backend": "llm", "candidates": len(candidates)})
        return results, completion.usage


def build_reranker(settings: Settings, model=None) -> Reranker:
    """Instantiate the reranker named by ``settings.reranker``."""
    backend = settings.reranker
    if backend == "none":
        return NullReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker(settings)
    if backend == "llm":
        if model is None:
            raise ConfigError("RERANKER=llm requires a chat model.")
        return LLMReranker(model)
    raise ConfigError(f"Unknown RERANKER {backend!r}; expected 'none', 'cross-encoder' or 'llm'.")
