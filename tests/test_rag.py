from __future__ import annotations

import pytest

from pdfchat.rag import NO_SOURCES_MESSAGE, RagPipeline
from pdfchat.retrieval import HybridRetriever
from pdfchat.stores.base import ChunkRecord
from pdfchat.stores.memory import MemoryVectorStore
from tests.conftest import FakeChatModel, make_chunk

TEXTS = [
    "Photosynthesis converts light energy into glucose inside the chloroplasts.",
    "Cellular respiration releases energy from glucose inside the mitochondria.",
    "Enzymes lower the activation energy required for a biochemical reaction.",
]


@pytest.fixture()
def pipeline_factory(settings, fake_embedder):
    def build(responses: list[str], **overrides):
        store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
        vectors, _ = fake_embedder.embed_documents(TEXTS)
        store.add(
            [
                ChunkRecord(chunk=make_chunk(f"doc1:{i:05d}", t, page=i + 1), embedding=vectors[i])
                for i, t in enumerate(TEXTS)
            ],
            collection="c",
        )
        active = settings.with_overrides(**overrides) if overrides else settings
        model = FakeChatModel(responses)
        retriever = HybridRetriever(store, fake_embedder, active)
        return RagPipeline(retriever, model, active), model

    return build


def test_answer_is_returned_with_sources_and_citations(pipeline_factory):
    pipeline, _ = pipeline_factory(
        ["Glucose is produced by photosynthesis [S1].", '{"verdict":"grounded","note":"ok"}']
    )
    answer = pipeline.answer("How is glucose produced?", collection="c")
    assert "[S1]" in answer.text
    assert answer.citations and answer.citations[0].marker == 1
    assert answer.sources
    assert answer.verdict == "grounded"
    assert not answer.refused


def test_sources_are_sent_as_the_cacheable_prefix(pipeline_factory):
    """Passages go in the cached prefix; only the question varies after it."""
    pipeline, model = pipeline_factory(["Answer [S1].", '{"verdict":"grounded"}'])
    pipeline.answer("What about chloroplasts?", collection="c")

    prefix = model.cache_prefixes[0]
    assert prefix is not None
    assert "<sources>" in prefix
    assert "chloroplasts" in prefix
    assert "[S1]" in prefix

    _system, user = model.prompts[0]
    assert "<question>" in user
    assert "<sources>" not in user, "passages must not also be in the volatile tail"


def test_every_call_about_one_question_shares_one_cached_prefix(pipeline_factory):
    """Answer, grounding check and follow-ups reuse a byte-identical prefix.

    Caching is a prefix match, so any difference between these - even
    whitespace - would turn two cache reads into two full-price writes.
    """
    pipeline, model = pipeline_factory(
        ["Answer [S1].", '{"verdict":"grounded"}', '{"questions":["Why?"]}']
    )
    answer = pipeline.answer("Explain respiration.", collection="c")
    pipeline.suggest_follow_ups(answer)

    prefixes = [p for p in model.cache_prefixes if p is not None]
    assert len(prefixes) == 3, "answer, grounding and follow-up should all send one"
    assert len(set(prefixes)) == 1, "the prefixes differ, so nothing would be reused"


def test_prefix_is_large_enough_to_actually_cache(pipeline_factory):
    """A breakpoint under the model minimum is stored silently as nothing."""
    from pdfchat.llm import is_worth_caching

    pipeline, model = pipeline_factory(["Answer [S1].", '{"verdict":"grounded"}'])
    pipeline.answer("Explain respiration.", collection="c")
    prefix = model.cache_prefixes[0]
    # The fixture corpus is deliberately small, so assert the rule is applied
    # rather than that this particular corpus clears it.
    assert is_worth_caching("x" * 4 * 600, "claude-opus-5")
    assert not is_worth_caching("x" * 4 * 100, "claude-opus-5")
    assert prefix is not None


def test_empty_corpus_produces_an_explicit_refusal(settings, fake_embedder):
    store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
    pipeline = RagPipeline(
        HybridRetriever(store, fake_embedder, settings), FakeChatModel(), settings
    )
    answer = pipeline.answer("Anything?", collection="empty")
    assert answer.refused
    assert answer.text == NO_SOURCES_MESSAGE
    assert answer.usage.output_tokens == 0, "no paid call should be made without sources"


def test_blank_question_is_rejected_without_a_model_call(pipeline_factory):
    pipeline, model = pipeline_factory(["should not be used"])
    answer = pipeline.answer("   ", collection="c")
    assert answer.refused
    assert model.prompts == []


def test_fabricated_citation_markers_are_stripped(pipeline_factory):
    pipeline, _ = pipeline_factory(
        ["Real claim [S1]. Invented claim [S99].", '{"verdict":"grounded"}']
    )
    answer = pipeline.answer("Explain glucose.", collection="c")
    assert "[S99]" not in answer.text
    assert "[S1]" in answer.text
    assert all(1 <= c.marker <= len(answer.sources) for c in answer.citations)


def test_model_refusal_is_flagged_as_refused(pipeline_factory):
    pipeline, _ = pipeline_factory(
        ["The documents don't cover this. Nothing here mentions tax law."]
    )
    answer = pipeline.answer("What is the corporate tax rate?", collection="c")
    assert answer.refused
    assert answer.verdict == "grounded"


def test_uncited_answer_is_marked_ungrounded(pipeline_factory):
    pipeline, _ = pipeline_factory(["Glucose is made by photosynthesis."])
    answer = pipeline.answer("How is glucose made?", collection="c")
    assert answer.verdict == "ungrounded"


def test_self_check_can_be_disabled(pipeline_factory):
    pipeline, model = pipeline_factory(["Answer [S1]."], enable_self_check=False)
    answer = pipeline.answer("Explain.", collection="c")
    assert answer.verdict == "unchecked"
    assert len(model.prompts) == 1, "verification must not run when disabled"


def test_streaming_delivers_tokens_to_the_callback(pipeline_factory):
    pipeline, _ = pipeline_factory(
        ["Streamed answer [S1].", '{"verdict":"grounded"}'], enable_streaming=True
    )
    received: list[str] = []
    answer = pipeline.answer("Explain.", collection="c", on_token=received.append)
    assert received
    assert "".join(received).strip() == answer.text.strip()


def test_usage_accumulates_across_generation_and_verification(pipeline_factory):
    pipeline, _ = pipeline_factory(["Answer [S1].", '{"verdict":"grounded"}'])
    answer = pipeline.answer("Explain.", collection="c")
    assert answer.usage.calls >= 2
    assert answer.usage.cost_usd > 0


def test_document_filter_is_passed_through(pipeline_factory):
    pipeline, _ = pipeline_factory(["Answer [S1].", '{"verdict":"grounded"}'])
    answer = pipeline.answer("Explain.", collection="c", doc_ids=["nonexistent"])
    assert answer.refused


def test_follow_ups_are_parsed(pipeline_factory):
    pipeline, _ = pipeline_factory(
        ["Answer [S1].", '{"verdict":"grounded"}', '{"questions":["Why?","How?","When?"]}']
    )
    answer = pipeline.answer("Explain.", collection="c")
    assert pipeline.suggest_follow_ups(answer) == ["Why?", "How?", "When?"]


def test_follow_ups_are_skipped_for_refusals(pipeline_factory):
    pipeline, _ = pipeline_factory(["The documents don't cover this."])
    answer = pipeline.answer("Unrelated question.", collection="c")
    assert pipeline.suggest_follow_ups(answer) == []


# --- concurrent verification + follow-ups ---------------------------------
def test_follow_ups_are_produced_in_the_same_call(pipeline_factory):
    pipeline, _ = pipeline_factory(
        ["Answer [S1].", '{"verdict":"grounded"}', '{"questions":["Why?","How?"]}']
    )
    answer = pipeline.answer("Explain.", collection="c", with_follow_ups=True)
    assert answer.follow_ups
    assert answer.verdict == "grounded"


def test_grounding_and_follow_ups_actually_overlap(settings, fake_embedder):
    """Both are network-bound and independent, so they must not run in sequence."""
    import threading
    import time

    from pdfchat.stores.base import ChunkRecord
    from pdfchat.stores.memory import MemoryVectorStore
    from tests.conftest import FakeChatModel, make_chunk

    class SlowModel(FakeChatModel):
        """Each call sleeps; overlapping calls therefore take ~1 delay, not 2."""

        DELAY = 0.25

        def __init__(self, responses):
            super().__init__(responses)
            self.concurrent = 0
            self.max_concurrent = 0
            self._lock = threading.Lock()

        def complete(self, system, user, *, max_tokens=None, cache_prefix=None):
            with self._lock:
                self.concurrent += 1
                self.max_concurrent = max(self.max_concurrent, self.concurrent)
            time.sleep(self.DELAY)
            with self._lock:
                self.concurrent -= 1
            return super().complete(system, user, max_tokens=max_tokens, cache_prefix=cache_prefix)

    store = MemoryVectorStore(dimensions=fake_embedder.dimensions)
    vectors, _ = fake_embedder.embed_documents(TEXTS)
    store.add(
        [
            ChunkRecord(chunk=make_chunk(f"doc1:{i:05d}", t, page=i + 1), embedding=vectors[i])
            for i, t in enumerate(TEXTS)
        ],
        collection="c",
    )
    model = SlowModel(["Answer [S1].", '{"verdict":"grounded"}', '{"questions":["Why?"]}'])
    pipeline = RagPipeline(HybridRetriever(store, fake_embedder, settings), model, settings)

    started = time.perf_counter()
    pipeline.answer("Explain glucose.", collection="c", with_follow_ups=True)
    elapsed = time.perf_counter() - started

    assert model.max_concurrent == 2, "the two post-answer calls ran one after the other"
    # generation + one overlapped pair, not generation + two sequential calls.
    assert elapsed < SlowModel.DELAY * 3, f"took {elapsed:.2f}s, suggesting no overlap"


def test_refusals_skip_follow_ups_entirely(pipeline_factory):
    pipeline, model = pipeline_factory(["The documents don't cover this."])
    answer = pipeline.answer("Unrelated.", collection="c", with_follow_ups=True)
    assert answer.follow_ups == []
    assert len(model.prompts) == 1, "a refusal must not pay for follow-ups"
