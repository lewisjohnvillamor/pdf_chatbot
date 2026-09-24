"""Embedding backends behind one interface.

Three implementations cover the deployment spectrum: a hosted OpenAI model,
a local sentence-transformers model for air-gapped or zero-cost use, and a
null backend that turns the system into pure BM25 retrieval so the app still
works with no embedding credentials at all.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from .config import Settings
from .errors import EmbeddingError
from .models import Usage
from .providers import build_openai_client, describe_sdk_error

logger = logging.getLogger(__name__)

#: OpenAI rejects oversized batches; this stays well inside the request limit.
_OPENAI_BATCH = 96


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-norm vectors."""

    name: str
    dimensions: int
    #: True when a call is a network round trip, so batches are worth issuing
    #: concurrently. False for on-device models, where concurrency only fights
    #: the GIL and the CPU - the same lesson as PDF extraction, except here
    #: processes are impractical because the model would be loaded per worker.
    is_remote: bool

    def embed_documents(self, texts: list[str]) -> tuple[np.ndarray, Usage]: ...

    def embed_query(self, text: str) -> tuple[np.ndarray, Usage]: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so dot product equals cosine similarity."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class NullEmbedder:
    """No-op backend: retrieval degrades to lexical-only, which still works."""

    name = "none"
    dimensions = 0
    is_remote = False

    def embed_documents(self, texts: list[str]) -> tuple[np.ndarray, Usage]:
        return np.zeros((len(texts), 0), dtype=np.float32), Usage()

    def embed_query(self, text: str) -> tuple[np.ndarray, Usage]:
        return np.zeros((1, 0), dtype=np.float32), Usage()


class OpenAIEmbedder:
    """Hosted embeddings via the OpenAI embeddings endpoint."""

    is_remote = True

    _DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, settings: Settings):
        if not settings.openai_api_key and not settings.openai_base_url:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY. Set it, point "
                "OPENAI_BASE_URL at a self-hosted endpoint, or switch "
                "EMBEDDING_PROVIDER to 'local' or 'none'."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise EmbeddingError("The 'openai' package is not installed.") from exc

        self._sdk = openai
        self._client = build_openai_client(settings)
        self._settings = settings
        self.name = settings.embedding_model
        self.dimensions = self._DIMENSIONS.get(settings.embedding_model, 1536)

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, Usage]:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32), Usage()
        vectors: list[list[float]] = []
        tokens = 0
        calls = 0
        for start in range(0, len(texts), _OPENAI_BATCH):
            batch = [t if t.strip() else " " for t in texts[start : start + _OPENAI_BATCH]]
            try:
                response = self._client.embeddings.create(model=self.name, input=batch)
            except Exception as exc:
                # Same ladder the chat backends use, so an auth or rate-limit
                # failure reads the same whichever call hit it first.
                raise EmbeddingError(
                    describe_sdk_error(
                        self._sdk,
                        exc,
                        provider="OpenAI",
                        model=self.name,
                        key_env="OPENAI_API_KEY",
                    )
                ) from exc
            vectors.extend(item.embedding for item in response.data)
            tokens += getattr(response.usage, "total_tokens", 0) or 0
            calls += 1
        price = self._settings.embedding_price(self.name)
        usage = Usage(embedding_tokens=tokens, cost_usd=tokens / 1_000_000 * price, calls=calls)
        return l2_normalize(np.asarray(vectors, dtype=np.float32)), usage

    def embed_documents(self, texts: list[str]) -> tuple[np.ndarray, Usage]:
        return self._embed(texts)

    def embed_query(self, text: str) -> tuple[np.ndarray, Usage]:
        return self._embed([text])


class LocalEmbedder:
    """On-device embeddings via sentence-transformers. No API key, no cost."""

    is_remote = False

    def __init__(self, settings: Settings):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=local requires sentence-transformers. "
                "Install it with: pip install -r requirements-local.txt"
            ) from exc
        self.name = settings.local_embedding_model
        self._model = SentenceTransformer(self.name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> tuple[np.ndarray, Usage]:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32), Usage()
        return self._encode(texts), Usage(calls=1)

    def embed_query(self, text: str) -> tuple[np.ndarray, Usage]:
        return self._encode([text]), Usage(calls=1)


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the embedder named by ``settings.embedding_provider``."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedding_provider == "local":
        return LocalEmbedder(settings)
    logger.warning("embeddings_disabled", extra={"reason": "EMBEDDING_PROVIDER=none"})
    return NullEmbedder()
