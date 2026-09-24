"""Environment-driven configuration.

The whole app reads its settings from one immutable :class:`Settings` object
built by :func:`load_settings`. Nothing else in the codebase touches
``os.environ``, which keeps the modules importable and testable without a
populated environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .errors import ConfigError

#: pg_table is interpolated into DDL/DML, so it is restricted to a bare identifier.
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

ChatProvider = Literal["anthropic", "openai"]
EmbeddingProvider = Literal["openai", "local", "none"]
VectorStoreBackend = Literal["memory", "postgres"]
PgIndexMethod = Literal["hnsw", "ivfflat"]
RerankerBackend = Literal["none", "cross-encoder", "llm"]

#: Chat models we know how to price, in USD per 1M tokens (input, output).
#: Override or extend at runtime with PDFCHAT_PRICE_OVERRIDES, e.g.
#: ``PDFCHAT_PRICE_OVERRIDES="my-model:1.5/7.5"``.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    # OpenAI (approximate list prices; override via env if yours differ)
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

#: Embedding models priced per 1M input tokens.
EMBEDDING_PRICES_USD_PER_MTOK: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

DEFAULT_CHAT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4.1-mini",
}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_price_overrides(raw: str) -> dict[str, tuple[float, float]]:
    """Parse ``"model:in/out,model2:in/out"`` into a price table."""
    overrides: dict[str, tuple[float, float]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            model, prices = entry.split(":", 1)
            input_price, output_price = prices.split("/", 1)
            overrides[model.strip()] = (float(input_price), float(output_price))
        except ValueError as exc:
            raise ConfigError(
                f"Invalid PDFCHAT_PRICE_OVERRIDES entry {entry!r}; "
                "expected 'model-name:INPUT/OUTPUT' in USD per 1M tokens"
            ) from exc
    return overrides


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration."""

    # --- providers -------------------------------------------------------
    chat_provider: ChatProvider = "anthropic"
    chat_model: str = "claude-opus-5"
    embedding_provider: EmbeddingProvider = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    #: Point at any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, a
    #: gateway) to keep document text inside your own infrastructure.
    openai_base_url: str | None = None

    # --- generation ------------------------------------------------------
    max_output_tokens: int = 4000
    effort: str = "medium"
    request_timeout_s: float = 120.0
    max_retries: int = 3

    # --- ingestion -------------------------------------------------------
    max_upload_mb: int = 50
    max_pages_per_doc: int = 2000
    min_chars_per_page_for_text_layer: int = 40

    # --- chunking --------------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- retrieval -------------------------------------------------------
    top_k: int = 6
    candidate_k: int = 30
    mmr_lambda: float = 0.6
    hybrid_dense_weight: float = 0.5
    #: Precision stage applied to the fused candidates before top-k.
    reranker: RerankerBackend = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    #: Candidates handed to the reranker. Larger means better recall into the
    #: precision stage, at a linear cost in reranking time.
    rerank_candidates: int = 20

    # --- storage ---------------------------------------------------------
    vector_store: VectorStoreBackend = "memory"
    database_url: str | None = None
    pg_table: str = "pdfchat_chunks"
    pg_pool_size: int = 8
    pg_ivfflat_lists: int = 100
    #: HNSW builds on an empty table and gives better recall than IVFFlat at a
    #: comparable query speed; IVFFlat remains for pgvector older than 0.5.0.
    pg_index_method: PgIndexMethod = "hnsw"
    persist_conversations: bool = True

    # --- access control --------------------------------------------------
    app_password_hash: str | None = None
    session_ttl_minutes: int = 720
    rate_limit_questions_per_hour: int = 120

    # --- behaviour -------------------------------------------------------
    enable_self_check: bool = True
    enable_streaming: bool = True
    #: Whether the sidebar shows the project's support link. Anyone hosting
    #: this for their own users can turn it off with SHOW_SUPPORT_LINK=false.
    show_support_link: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".pdfchat_cache"))
    log_level: str = "INFO"
    price_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def chat_price(self, model: str | None = None) -> tuple[float, float]:
        """USD per 1M (input, output) tokens for ``model``; zeros if unknown."""
        name = model or self.chat_model
        if name in self.price_overrides:
            return self.price_overrides[name]
        return MODEL_PRICES_USD_PER_MTOK.get(name, (0.0, 0.0))

    def embedding_price(self, model: str | None = None) -> float:
        """USD per 1M input tokens for the embedding model; 0.0 if unknown/local."""
        name = model or self.embedding_model
        if name in self.price_overrides:
            return self.price_overrides[name][0]
        return EMBEDDING_PRICES_USD_PER_MTOK.get(name, 0.0)

    def validated(self) -> Settings:
        """Return self after checking cross-field invariants."""
        if self.chat_provider not in ("anthropic", "openai"):
            raise ConfigError(f"Unknown CHAT_PROVIDER {self.chat_provider!r}")
        if self.embedding_provider not in ("openai", "local", "none"):
            raise ConfigError(f"Unknown EMBEDDING_PROVIDER {self.embedding_provider!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k > self.candidate_k:
            raise ConfigError("TOP_K must not exceed CANDIDATE_K")
        if self.reranker not in ("none", "cross-encoder", "llm"):
            raise ConfigError(
                f"Unknown RERANKER {self.reranker!r}; expected 'none', 'cross-encoder' or 'llm'."
            )
        if self.reranker != "none" and self.rerank_candidates < self.top_k:
            raise ConfigError("RERANK_CANDIDATES must be at least TOP_K")
        if self.vector_store not in ("memory", "postgres"):
            raise ConfigError(f"Unknown VECTOR_STORE {self.vector_store!r}")
        if self.vector_store == "postgres":
            if not self.database_url:
                raise ConfigError("VECTOR_STORE=postgres requires DATABASE_URL")
            if self.embedding_provider == "none":
                raise ConfigError(
                    "VECTOR_STORE=postgres needs embeddings; set EMBEDDING_PROVIDER "
                    "to 'openai' or 'local'."
                )
        if self.pg_index_method not in ("hnsw", "ivfflat"):
            raise ConfigError(
                f"Unknown PG_INDEX_METHOD {self.pg_index_method!r}; expected 'hnsw' or 'ivfflat'."
            )
        if not _TABLE_NAME_RE.match(self.pg_table):
            # The table name is interpolated into SQL, so it must be a plain
            # identifier — never accept arbitrary text here.
            raise ConfigError(f"PG_TABLE {self.pg_table!r} must match [A-Za-z_][A-Za-z0-9_]*")
        return self

    @property
    def auth_enabled(self) -> bool:
        return bool(self.app_password_hash)

    def missing_credentials(self) -> list[str]:
        """Names of API keys required by the current provider selection."""
        missing: list[str] = []
        if self.chat_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        # A self-hosted OpenAI-compatible endpoint often needs no real key, so
        # only require one when talking to OpenAI itself.
        needs_key = not self.openai_base_url
        if self.chat_provider == "openai" and not self.openai_api_key and needs_key:
            missing.append("OPENAI_API_KEY")
        if self.embedding_provider == "openai" and not self.openai_api_key and needs_key:
            missing.append("OPENAI_API_KEY (for embeddings)")
        return missing

    def with_overrides(self, **kwargs: object) -> Settings:
        """Return a copy with fields replaced — used by the UI's live controls."""
        return replace(self, **kwargs).validated()  # type: ignore[arg-type]


def load_settings() -> Settings:
    """Build :class:`Settings` from the process environment."""
    provider = _env_str("CHAT_PROVIDER", "anthropic").lower()
    default_model = DEFAULT_CHAT_MODELS.get(provider, "claude-opus-5")
    overrides_raw = os.getenv("PDFCHAT_PRICE_OVERRIDES", "")

    settings = Settings(
        chat_provider=provider,  # type: ignore[arg-type]
        chat_model=_env_str("CHAT_MODEL", default_model),
        embedding_provider=_env_str("EMBEDDING_PROVIDER", "openai").lower(),  # type: ignore[arg-type]
        embedding_model=_env_str("EMBEDDING_MODEL", "text-embedding-3-small"),
        local_embedding_model=_env_str(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 4000, minimum=256),
        effort=_env_str("EFFORT", "medium").lower(),
        request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 120.0, minimum=5.0, maximum=900.0),
        max_retries=_env_int("MAX_RETRIES", 3, minimum=0),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
        max_pages_per_doc=_env_int("MAX_PAGES_PER_DOC", 2000),
        chunk_size=_env_int("CHUNK_SIZE", 1200, minimum=200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200, minimum=0),
        top_k=_env_int("TOP_K", 6),
        candidate_k=_env_int("CANDIDATE_K", 30),
        mmr_lambda=_env_float("MMR_LAMBDA", 0.6, minimum=0.0, maximum=1.0),
        hybrid_dense_weight=_env_float("HYBRID_DENSE_WEIGHT", 0.5, minimum=0.0, maximum=1.0),
        reranker=_env_str("RERANKER", "none").lower(),  # type: ignore[arg-type]
        reranker_model=_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_candidates=_env_int("RERANK_CANDIDATES", 20),
        enable_self_check=_env_bool("ENABLE_SELF_CHECK", True),
        enable_streaming=_env_bool("ENABLE_STREAMING", True),
        show_support_link=_env_bool("SHOW_SUPPORT_LINK", True),
        vector_store=_env_str("VECTOR_STORE", "memory").lower(),  # type: ignore[arg-type]
        database_url=os.getenv("DATABASE_URL") or None,
        pg_table=_env_str("PG_TABLE", "pdfchat_chunks"),
        pg_pool_size=_env_int("PG_POOL_SIZE", 8),
        pg_ivfflat_lists=_env_int("PG_IVFFLAT_LISTS", 100),
        pg_index_method=_env_str("PG_INDEX_METHOD", "hnsw").lower(),  # type: ignore[arg-type]
        persist_conversations=_env_bool("PERSIST_CONVERSATIONS", True),
        app_password_hash=os.getenv("APP_PASSWORD_HASH") or None,
        session_ttl_minutes=_env_int("SESSION_TTL_MINUTES", 720),
        rate_limit_questions_per_hour=_env_int("RATE_LIMIT_QUESTIONS_PER_HOUR", 120),
        cache_dir=Path(_env_str("CACHE_DIR", ".pdfchat_cache")),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        price_overrides=_parse_price_overrides(overrides_raw) if overrides_raw else {},
    )
    return settings.validated()
