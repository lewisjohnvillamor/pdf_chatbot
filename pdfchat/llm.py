"""Chat providers behind one interface, with streaming and cost accounting.

Both backends expose the same three operations — ``complete``, ``stream`` and
``complete_json`` — so the rest of the app never branches on provider. The
Anthropic path additionally uses prompt caching: retrieved sources are the
bulk of every request and are reused across turns, so caching them cuts both
latency and cost substantially on multi-turn study sessions.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .config import Settings
from .errors import ProviderError
from .models import Usage

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Models are asked for bare JSON, but occasionally wrap it in a fenced block
    or add a sentence of preamble. Rather than fail the whole study-tool
    request, peel those layers off before giving up.
    """
    text = (text or "").strip()
    for candidate in (text, *(m.group(1) for m in _JSON_BLOCK.finditer(text))):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    match = _BARE_OBJECT.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ProviderError(
        "The model did not return valid JSON for this request. Try again, or "
        "reduce the number of items requested."
    )


@dataclass(slots=True)
class Completion:
    """A finished generation plus its usage."""

    text: str
    usage: Usage
    stop_reason: str | None = None


@runtime_checkable
class ChatModel(Protocol):
    """A chat completion backend."""

    name: str
    provider: str

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> Completion: ...

    def stream(self, system: str, user: str, *, max_tokens: int | None = None) -> Iterator[str]: ...

    def last_usage(self) -> Usage: ...


def _cost(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = settings.chat_price(model)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class AnthropicChat:
    """Claude via the Anthropic Messages API."""

    provider = "anthropic"

    def __init__(self, settings: Settings):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError("The 'anthropic' package is not installed.") from exc
        if not settings.anthropic_api_key:
            raise ProviderError(
                "CHAT_PROVIDER=anthropic requires ANTHROPIC_API_KEY. Set it in your "
                "environment or .env file."
            )
        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
        self._settings = settings
        self.name = settings.chat_model
        self._last = Usage()

    # ------------------------------------------------------------------
    def _request_kwargs(self, system: str, user: str, max_tokens: int | None) -> dict[str, Any]:
        return {
            "model": self.name,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            # The system prompt is byte-stable across turns, so caching it is free
            # recall on every follow-up question in a session.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._settings.effort},
        }

    def _record(self, usage: Any) -> Usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Cache reads bill at ~10% and writes at ~125% of the input rate.
        billable_input = input_tokens + cache_write * 1.25 + cache_read * 0.1
        recorded = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=_cost(self._settings, self.name, int(billable_input), output_tokens),
            calls=1,
        )
        self._last = recorded
        return recorded

    def _translate(self, exc: Exception) -> ProviderError:
        sdk = self._sdk
        if isinstance(exc, sdk.AuthenticationError):
            return ProviderError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.")
        if isinstance(exc, sdk.RateLimitError):
            return ProviderError("Rate limited by Anthropic. Wait a moment and try again.")
        if isinstance(exc, sdk.NotFoundError):
            return ProviderError(f"Model {self.name!r} is not available to this account.")
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderError("Could not reach the Anthropic API. Check network access.")
        if isinstance(exc, sdk.APIStatusError):
            return ProviderError(f"Anthropic API error ({exc.status_code}): {exc.message}")
        return ProviderError(f"Anthropic request failed: {exc}")

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> Completion:
        try:
            # Streaming under the hood keeps large max_tokens from hitting the
            # SDK's HTTP timeout, while still returning one finished message.
            with self._client.messages.stream(
                **self._request_kwargs(system, user, max_tokens)
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            raise self._translate(exc) from exc

        if message.stop_reason == "refusal":
            raise ProviderError("The model declined to answer this request. Rephrase the question.")
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(
            text=text, usage=self._record(message.usage), stop_reason=message.stop_reason
        )

    def stream(self, system: str, user: str, *, max_tokens: int | None = None) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                **self._request_kwargs(system, user, max_tokens)
            ) as stream:
                yield from stream.text_stream
                self._record(stream.get_final_message().usage)
        except Exception as exc:
            raise self._translate(exc) from exc

    def last_usage(self) -> Usage:
        return self._last


class OpenAIChat:
    """GPT models via the OpenAI Chat Completions API."""

    provider = "openai"

    def __init__(self, settings: Settings):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ProviderError("The 'openai' package is not installed.") from exc
        if not settings.openai_api_key and not settings.openai_base_url:
            raise ProviderError(
                "CHAT_PROVIDER=openai requires OPENAI_API_KEY. Set it in your "
                "environment or .env file, or set OPENAI_BASE_URL to point at a "
                "self-hosted OpenAI-compatible endpoint."
            )
        self._sdk = openai
        self._client = openai.OpenAI(
            # Local gateways accept any non-empty key.
            api_key=settings.openai_api_key or "not-needed",
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
        self._settings = settings
        self.name = settings.chat_model
        self._last = Usage()

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _record(self, usage: Any) -> Usage:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        recorded = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost(self._settings, self.name, input_tokens, output_tokens),
            calls=1,
        )
        self._last = recorded
        return recorded

    def _translate(self, exc: Exception) -> ProviderError:
        sdk = self._sdk
        if isinstance(exc, sdk.AuthenticationError):
            return ProviderError("OpenAI rejected the API key. Check OPENAI_API_KEY.")
        if isinstance(exc, sdk.RateLimitError):
            return ProviderError("Rate limited by OpenAI. Wait a moment and try again.")
        if isinstance(exc, sdk.NotFoundError):
            return ProviderError(f"Model {self.name!r} is not available to this account.")
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderError("Could not reach the OpenAI API. Check network access.")
        if isinstance(exc, sdk.APIStatusError):
            return ProviderError(f"OpenAI API error ({exc.status_code}): {exc.message}")
        return ProviderError(f"OpenAI request failed: {exc}")

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> Completion:
        try:
            response = self._client.chat.completions.create(
                model=self.name,
                max_tokens=max_tokens or self._settings.max_output_tokens,
                messages=self._messages(system, user),
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        choice = response.choices[0]
        return Completion(
            text=choice.message.content or "",
            usage=self._record(response.usage),
            stop_reason=choice.finish_reason,
        )

    def stream(self, system: str, user: str, *, max_tokens: int | None = None) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.name,
                max_tokens=max_tokens or self._settings.max_output_tokens,
                messages=self._messages(system, user),
                stream=True,
                stream_options={"include_usage": True},
            )
            for event in stream:
                if getattr(event, "usage", None):
                    self._record(event.usage)
                for choice in event.choices or []:
                    piece = choice.delta.content if choice.delta else None
                    if piece:
                        yield piece
        except Exception as exc:
            raise self._translate(exc) from exc

    def last_usage(self) -> Usage:
        return self._last


def build_chat_model(settings: Settings) -> ChatModel:
    """Instantiate the chat backend named by ``settings.chat_provider``."""
    if settings.chat_provider == "anthropic":
        return AnthropicChat(settings)
    return OpenAIChat(settings)


def complete_json(
    model: ChatModel, system: str, user: str, *, max_tokens: int | None = None
) -> tuple[dict[str, Any], Usage]:
    """Run a completion whose response is expected to be a JSON object."""
    completion = model.complete(system, user, max_tokens=max_tokens)
    return extract_json(completion.text), completion.usage
