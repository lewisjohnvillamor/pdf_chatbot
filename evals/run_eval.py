"""Measure retrieval quality against the labelled gold set.

    python evals/run_eval.py                 # offline, hashed embeddings
    python evals/run_eval.py --real-embeddings   # uses EMBEDDING_PROVIDER

The corpus is rendered to real PDFs and pushed through the whole production
path — ingest, clean, chunk, embed, index — so the numbers reflect the system
that actually runs, not a shortcut around it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdfchat.config import load_settings
from pdfchat.evaluation import (
    EvalResult,
    configurations_are_indistinguishable,
    evaluate,
    format_table,
    load_goldset,
    sweep_dense_weight,
)
from pdfchat.indexing import build_index
from pdfchat.models import Usage
from pdfchat.retrieval import HybridRetriever
from pdfchat.stores.memory import MemoryVectorStore

CORPUS_DIR = Path(__file__).parent / "corpus"
GOLDSET = Path(__file__).parent / "goldset.json"
COLLECTION = "eval"


class HashEmbedder:
    """Deterministic hashed bag-of-words vectors.

    Lets the harness run offline and in CI with zero API cost. It captures
    exact-term overlap but not synonymy, so it *understates* what a real
    embedding model contributes — dense and hybrid rows are a lower bound.
    Use --real-embeddings for the numbers you would actually ship on.
    """

    name = "hash-bow"
    dimensions = 512

    def _vector(self, text: str) -> np.ndarray:
        from pdfchat.lexical import tokenize

        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in tokenize(text):
            slot = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
            vector[slot % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32), Usage()
        return np.vstack([self._vector(t) for t in texts]), Usage()

    def embed_query(self, text):
        return self._vector(text).reshape(1, -1), Usage()


class TextUpload:
    """Renders a .txt fixture to a real PDF so ingestion runs for real."""

    def __init__(self, path: Path):
        self.name = f"{path.stem}.pdf"
        self._path = path

    def getvalue(self) -> bytes:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=LETTER)
        lines = self._path.read_text(encoding="utf-8").splitlines()
        title = lines[0] if lines else self._path.stem
        page_number, y = 1, 700
        # A running header and a page number on every page, exactly like a real
        # document — so the cleaning pipeline is exercised, not bypassed.
        pdf.drawString(72, 740, title)
        for line in lines[1:]:
            if y < 90:
                pdf.drawString(72, 60, str(page_number))
                pdf.showPage()
                page_number += 1
                y = 700
                pdf.drawString(72, 740, title)
            pdf.drawString(72, y, line)
            y -= 15
        pdf.drawString(72, 60, str(page_number))
        pdf.save()
        return buffer.getvalue()


def index_corpus(uploads, settings, embedder):
    """Build a fresh index for one configuration."""
    store = MemoryVectorStore(dimensions=embedder.dimensions)
    report = build_index(
        uploads, settings=settings, embedder=embedder, store=store, collection=COLLECTION
    )
    if not report.succeeded:
        print(f"Indexing failed: {report.problems}", file=sys.stderr)
        return store, None
    return store, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-embeddings",
        action="store_true",
        help="use the configured EMBEDDING_PROVIDER instead of offline hashing",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="characters per passage; smaller means finer-grained citations",
    )
    parser.add_argument(
        "--sweep-chunk-size",
        action="store_true",
        help="also compare chunk sizes at the best dense weight",
    )
    parser.add_argument("--verbose", action="store_true", help="list every missed question")
    args = parser.parse_args()

    settings = load_settings().with_overrides(
        top_k=args.top_k,
        candidate_k=30,
        chunk_size=args.chunk_size,
        chunk_overlap=min(args.chunk_size // 4, 200),
        cache_dir=None,
    )

    if args.real_embeddings:
        from pdfchat.embeddings import build_embedder

        embedder = build_embedder(settings)
        print(f"Embeddings: {embedder.name} ({embedder.dimensions}d, live API)\n")
    else:
        embedder = HashEmbedder()
        print(
            "Embeddings: offline hashed bag-of-words.\n"
            "  Captures exact-term overlap but NOT synonymy, so dense/hybrid rows\n"
            "  are a LOWER BOUND. Re-run with --real-embeddings before tuning.\n"
        )

    uploads = [TextUpload(p) for p in sorted(CORPUS_DIR.glob("*.txt"))]
    if not uploads:
        print(f"No corpus files in {CORPUS_DIR}", file=sys.stderr)
        return 1

    store, report = index_corpus(uploads, settings, embedder)
    if report is None:
        return 1
    print(
        f"Corpus: {len(report.documents)} documents, {report.page_count} pages, "
        f"{report.chunk_count} passages at chunk_size={settings.chunk_size}"
    )

    # An eval whose corpus is smaller than k is arithmetically incapable of
    # discriminating between configurations: every retriever returns everything
    # and scores 1.0. Refuse to print numbers that cannot mean anything.
    if report.chunk_count <= args.top_k * 2:
        print(
            f"\nERROR: only {report.chunk_count} passages for k={args.top_k}. Every "
            "configuration will score identically because there is nothing to "
            "discriminate between. Add corpus material or lower --top-k.",
            file=sys.stderr,
        )
        return 2

    cases = load_goldset(GOLDSET)
    negatives = sum(1 for case in cases if case.negative)
    print(
        f"Gold set: {len(cases) - negatives} scored questions "
        f"(+{negatives} negative control excluded from ranking metrics), k={args.top_k}\n"
    )

    results: list[EvalResult] = sweep_dense_weight(
        store, embedder, settings, cases, collection=COLLECTION
    )
    print(format_table(results))

    if configurations_are_indistinguishable(results):
        print(
            "\nFINDING: every configuration scored identically.\n"
            "  The dense and lexical rankers are returning the same order, so RRF\n"
            "  has nothing to fuse. This is expected with the offline hashed\n"
            "  embedder (it is bag-of-words, i.e. the same signal BM25 uses).\n"
            "  This run CANNOT tell you what to set HYBRID_DENSE_WEIGHT to.\n"
            "  Re-run with --real-embeddings to get a usable answer."
        )
        best = results[0]
    else:
        best = max(results, key=lambda r: (r.recall, r.mrr))
        print(f"\nBest configuration: {best.label}")

    if args.sweep_chunk_size:
        print("\nChunk-size sweep (at the best dense weight):")
        weight = best.dense_weight if best.dense_weight is not None else 0.5
        chunk_results = []
        for size in (250, 400, 700, 1200):
            tuned = settings.with_overrides(
                chunk_size=size, chunk_overlap=min(size // 4, 200), hybrid_dense_weight=weight
            )
            sized_store, sized_report = index_corpus(uploads, tuned, embedder)
            if sized_report is None:
                continue
            chunk_results.append(
                evaluate(
                    HybridRetriever(sized_store, embedder, tuned),
                    cases,
                    collection=COLLECTION,
                    k=tuned.top_k,
                    label=f"chunk={size} ({sized_report.chunk_count}p)",
                    corpus_size=sized_report.chunk_count,
                )
            )
        print(format_table(chunk_results))

    if best.misses:
        print(f"\nUnretrieved questions under the best configuration ({len(best.misses)}):")
        for miss in best.misses:
            print(f"  - {miss.case.question}")
            print(f"      wanted: {miss.case.must_contain[0]!r}  ({miss.case.note})")

    if args.verbose:
        for result in results:
            if result.misses:
                print(f"\n{result.label} missed:")
                for miss in result.misses:
                    print(f"  - {miss.case.question}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
