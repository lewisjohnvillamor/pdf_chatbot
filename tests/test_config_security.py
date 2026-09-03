from __future__ import annotations

import time

import pytest

from pdfchat.config import Settings, load_settings
from pdfchat.errors import ConfigError
from pdfchat.security import RateLimiter, hash_password, verify_password


def test_defaults_load_without_any_environment(monkeypatch):
    for key in ("CHAT_PROVIDER", "EMBEDDING_PROVIDER", "VECTOR_STORE", "CHUNK_SIZE"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings()
    assert settings.chat_provider == "anthropic"
    assert settings.chat_model == "claude-opus-5"


def test_provider_choice_selects_its_default_model(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "openai")
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    assert load_settings().chat_model == "gpt-4.1-mini"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "sorcery")
    with pytest.raises(ConfigError, match="Unknown CHAT_PROVIDER"):
        load_settings()


def test_non_numeric_setting_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "big")
    with pytest.raises(ConfigError, match="CHUNK_SIZE must be an integer"):
        load_settings()


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=500, chunk_overlap=500).validated()


def test_top_k_must_not_exceed_candidate_k():
    with pytest.raises(ConfigError, match="TOP_K"):
        Settings(top_k=50, candidate_k=10).validated()


def test_missing_credentials_are_reported_per_provider():
    anthropic = Settings(chat_provider="anthropic", embedding_provider="none")
    assert anthropic.missing_credentials() == ["ANTHROPIC_API_KEY"]
    openai = Settings(chat_provider="openai", embedding_provider="openai")
    assert len(openai.missing_credentials()) == 2


def test_price_lookup_and_overrides():
    settings = Settings(chat_model="claude-opus-5")
    assert settings.chat_price() == (5.00, 25.00)
    assert Settings(chat_model="unknown-model").chat_price() == (0.0, 0.0)
    custom = Settings(chat_model="my-model", price_overrides={"my-model": (1.0, 2.0)})
    assert custom.chat_price() == (1.0, 2.0)


def test_price_overrides_parse_from_env(monkeypatch):
    monkeypatch.setenv("PDFCHAT_PRICE_OVERRIDES", "my-model:1.5/7.5")
    assert load_settings().chat_price("my-model") == (1.5, 7.5)


def test_malformed_price_override_is_rejected(monkeypatch):
    monkeypatch.setenv("PDFCHAT_PRICE_OVERRIDES", "broken-entry")
    with pytest.raises(ConfigError, match="PDFCHAT_PRICE_OVERRIDES"):
        load_settings()


def test_with_overrides_returns_a_validated_copy():
    settings = Settings()
    assert settings.with_overrides(top_k=2).top_k == 2
    assert settings.top_k != 2, "the original must stay immutable"


def test_password_hash_roundtrip():
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_password_hash_is_salted():
    assert hash_password("same") != hash_password("same")


def test_malformed_hash_never_authenticates():
    assert not verify_password("anything", "garbage")
    assert not verify_password("anything", "")


def test_empty_password_is_rejected():
    with pytest.raises(ValueError):
        hash_password("")


def test_rate_limiter_blocks_past_the_limit():
    limiter = RateLimiter(max_events=3)
    assert [limiter.allow() for _ in range(5)] == [True, True, True, False, False]
    assert limiter.remaining == 0
    assert limiter.retry_after_seconds() > 0


def test_rate_limiter_of_zero_is_unlimited():
    limiter = RateLimiter(max_events=0)
    assert all(limiter.allow() for _ in range(100))
    assert limiter.remaining == -1


# --- session expiry --------------------------------------------------------
def test_session_expires_after_its_ttl():
    from pdfchat.security import session_expired

    now = time.time()
    assert not session_expired(now, ttl_minutes=60)
    assert not session_expired(now - 59 * 60, ttl_minutes=60)
    assert session_expired(now - 61 * 60, ttl_minutes=60)


def test_missing_timestamp_counts_as_expired():
    """A session that cannot prove when it authenticated is not trusted."""
    from pdfchat.security import session_expired

    assert session_expired(None, ttl_minutes=60)


def test_zero_ttl_disables_expiry():
    from pdfchat.security import session_expired

    assert not session_expired(None, ttl_minutes=0)
    assert not session_expired(time.time() - 10**6, ttl_minutes=0)


def test_session_ttl_is_configurable(monkeypatch):
    monkeypatch.setenv("SESSION_TTL_MINUTES", "30")
    assert load_settings().session_ttl_minutes == 30
