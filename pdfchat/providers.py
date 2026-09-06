"""Plumbing shared by the Anthropic and OpenAI backends.

The two SDKs expose the same exception class names and the same OpenAI-style
client constructor, so the chat and embedding backends were carrying two
near-identical copies of both. Divergent copies of error handling are how one
provider quietly ends up with worse messages than the other.
"""

from __future__ import annotations

from typing import Any

from .config import Settings


def describe_sdk_error(sdk: Any, exc: Exception, *, provider: str, model: str, key_env: str) -> str:
    """Turn an SDK exception into a message that says what to do next.

    ``sdk`` is the provider module itself: both ``anthropic`` and ``openai``
    define this same set of exception classes, so one ladder serves both.
    """
    if isinstance(exc, sdk.AuthenticationError):
        return f"{provider} rejected the API key. Check {key_env}."
    if isinstance(exc, sdk.RateLimitError):
        return f"Rate limited by {provider}. Wait a moment and try again."
    if isinstance(exc, sdk.NotFoundError):
        return f"Model {model!r} is not available to this account."
    if isinstance(exc, sdk.APIConnectionError):
        return f"Could not reach the {provider} API. Check network access."
    if isinstance(exc, sdk.APIStatusError):
        return f"{provider} API error ({exc.status_code}): {exc.message}"
    return f"{provider} request failed: {exc}"


def build_openai_client(settings: Settings) -> Any:
    """Construct an OpenAI client, honouring a self-hosted base URL.

    Used for both chat and embeddings, which is why it lives here rather than
    in either one.
    """
    from openai import OpenAI

    return OpenAI(
        # Self-hosted gateways accept any non-empty key.
        api_key=settings.openai_api_key or "not-needed",
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )
