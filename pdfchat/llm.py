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
from .providers import build_openai_client, describe_sdk_error

logger = logging.getLogger(__name__)

#: Minimum prefix Anthropic will cache, by model. Below this the API silently
#: stores nothing - no error, just cache_creation_input_tokens: 0 - so a
#: breakpoint on a short prefix is wasted markup that reads as if it works.
MIN_CACHEABLE_TOKENS: dict[str, int] = {
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-haiku-4-5": 4096,
}
#: Used for models not in the table. The highest published minimum, so an
#: unknown model never gets a breakpoint that silently does nothing.
DEFAULT_MIN_CACHEABLE_TOKENS = 4096


def is_worth_caching(text: str, model: str) -> bool:
    """Is ``text`` long enough that Anthropic will actually cache it?"""
    minimum = MIN_CACHEABLE_TOKENS.get(model, DEFAULT_MIN_CACHEABLE_TOKENS)
    return len(text) // 4 >= minimum


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

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
    ) -> Completion: ...

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
    ) -> Iterator[str]: ...

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
    def _request_kwargs(
        self,
        system: str,
        user: str,
        max_tokens: int | None,
        cache_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Build the request, placing a cache breakpoint where it can pay off.

        Caching is a prefix match over tools -> system -> messages, so a
        breakpoint on ``cache_prefix`` covers the system prompt too. The system
        prompt alone is far below every model's minimum, which is why marking
        it on its own cached nothing.
        """
        content: Any = user
        if cache_prefix and is_worth_caching(cache_prefix + system, self.name):
            content = [
                {
                    "type": "text",
                    "text": cache_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": user},
            ]
        elif cache_prefix:
            # Too short to cache: send it as plain text rather than pretend.
            content = f"{cache_prefix}\n{user}"

        return {
            "model": self.name,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
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
        return ProviderError(
            describe_sdk_error(
                self._sdk,
                exc,
                provider="Anthropic",
                model=self.name,
                key_env="ANTHROPIC_API_KEY",
            )
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
    ) -> Completion:
        try:
            # Streaming under the hood keeps large max_tokens from hitting the
            # SDK's HTTP timeout, while still returning one finished message.
            with self._client.messages.stream(
                **self._request_kwargs(system, user, max_tokens, cache_prefix)
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

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
    ) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                **self._request_kwargs(system, user, max_tokens, cache_prefix)
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
        self._client = build_openai_client(settings)
        self._settings = settings
        self.name = settings.chat_model
        self._last = Usage()

    def _messages(
        self, system: str, user: str, cache_prefix: str | None = None
    ) -> list[dict[str, str]]:
        # OpenAI caches long prefixes automatically with no request-side markup,
        # so the prefix only has to stay first and byte-stable.
        content = f"{cache_prefix}\n{user}" if cache_prefix else user
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

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
        return ProviderError(
            describe_sdk_error(
                self._sdk, exc, provider="OpenAI", model=self.name, key_env="OPENAI_API_KEY"
            )
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
    ) -> Completion:
        try:
            response = self._client.chat.completions.create(
                model=self.name,
                max_tokens=max_tokens or self._settings.max_output_tokens,
                messages=self._messages(system, user, cache_prefix),
            )
        except Exception as exc:
            raise self._translate(exc) from exc
        choice = response.choices[0]
        return Completion(
            text=choice.message.content or "",
            usage=self._record(response.usage),
            stop_reason=choice.finish_reason,
        )

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        cache_prefix: str | None = None,
    ) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.name,
                max_tokens=max_tokens or self._settings.max_output_tokens,
                messages=self._messages(system, user, cache_prefix),
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
    model: ChatModel,
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
    cache_prefix: str | None = None,
) -> tuple[dict[str, Any], Usage]:
    """Run a completion whose response is expected to be a JSON object."""
    completion = model.complete(system, user, max_tokens=max_tokens, cache_prefix=cache_prefix)
    return extract_json(completion.text), completion.usage
