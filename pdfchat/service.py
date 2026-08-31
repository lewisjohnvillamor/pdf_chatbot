"""Composition root: build every collaborator from one Settings object.

Keeping construction in one place means the Streamlit layer holds a single
object, and tests can assemble the same graph with fakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder, build_embedder
from .history import HistoryStore, build_history_store
from .indexing import IndexReport, build_index
from .ingest import UploadedFile
from .llm import ChatModel, build_chat_model
from .rag import RagPipeline
from .reranking import build_reranker
from .retrieval import HybridRetriever
from .stores import build_store
from .stores.base import VectorStore

logger = logging.getLogger(__name__)

#: Every chunk in a single-tenant deployment lives in one logical collection.
DEFAULT_COLLECTION = "default"


@dataclass(slots=True)
class ChatService:
    """The whole application graph, assembled and ready to use."""

    settings: Settings
    embedder: Embedder
    store: VectorStore
    model: ChatModel
    retriever: HybridRetriever
    pipeline: RagPipeline
    history: HistoryStore
    collection: str = DEFAULT_COLLECTION

    # ------------------------------------------------------------------
    def index(self, files: list[UploadedFile], progress=None) -> IndexReport:
        """Ingest uploads into the corpus and refresh the lexical index."""
        report = build_index(
            files,
            settings=self.settings,
            embedder=self.embedder,
            store=self.store,
            collection=self.collection,
            progress=progress,
        )
        self.retriever.invalidate()
        return report

    def documents(self) -> list[tuple[str, str, int]]:
        """``(doc_id, filename, chunk_count)`` for everything indexed."""
        return sorted(self.store.list_documents(self.collection), key=lambda row: row[1])

    def chunk_count(self) -> int:
        return self.store.count(self.collection)

    def reset(self) -> None:
        """Drop the corpus. Conversations are kept; the material is not."""
        self.store.delete_collection(self.collection)
        self.retriever.invalidate()

    def close(self) -> None:
        self.store.close()


def build_service(settings: Settings, *, collection: str = DEFAULT_COLLECTION) -> ChatService:
    """Construct the full service graph for ``settings``."""
    embedder = build_embedder(settings)
    store = build_store(settings, dimensions=embedder.dimensions)
    model = build_chat_model(settings)
    reranker = build_reranker(settings, model)
    retriever = HybridRetriever(store, embedder, settings, reranker=reranker)
    pipeline = RagPipeline(retriever, model, settings)
    history = build_history_store(settings)
    logger.info(
        "service_ready",
        extra={
            "chat_provider": settings.chat_provider,
            "chat_model": settings.chat_model,
            "embedding_provider": settings.embedding_provider,
            "vector_store": settings.vector_store,
            "reranker": settings.reranker,
            "dimensions": embedder.dimensions,
        },
    )
    return ChatService(
        settings=settings,
        embedder=embedder,
        store=store,
        model=model,
        retriever=retriever,
        pipeline=pipeline,
        history=history,
        collection=collection,
    )
