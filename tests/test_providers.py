"""Tests for the plumbing shared by both provider backends.

This ladder was previously duplicated per provider and tested through neither.
Now that one copy serves both, a mistake here degrades every error message in
the app at once, so it is worth testing directly.
"""

from __future__ import annotations

import anthropic
import openai
import pytest

from pdfchat.config import Settings
from pdfchat.providers import build_openai_client, describe_sdk_error

PROVIDERS = [
    pytest.param(anthropic, "Anthropic", "ANTHROPIC_API_KEY", id="anthropic"),
    pytest.param(openai, "OpenAI", "OPENAI_API_KEY", id="openai"),
]


def _make(sdk, kind: str):
    """Build a real SDK exception without touching the network."""
    request = object()
    if kind == "auth":
        return sdk.AuthenticationError("bad key", response=_Response(401), body=None)
    if kind == "rate":
        return sdk.RateLimitError("slow down", response=_Response(429), body=None)
    if kind == "notfound":
        return sdk.NotFoundError("no model", response=_Response(404), body=None)
    if kind == "conn":
        return sdk.APIConnectionError(request=request)
    raise AssertionError(kind)


class _Response:
    """The minimum an SDK error needs from an HTTP response."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        self.headers = {}
        self.request = object()


@pytest.mark.parametrize(("sdk", "label", "key_env"), PROVIDERS)
def test_auth_error_names_the_variable_to_fix(sdk, label, key_env):
    message = describe_sdk_error(
        sdk, _make(sdk, "auth"), provider=label, model="m", key_env=key_env
    )
    assert label in message
    assert key_env in message, "the message must say which variable to check"


@pytest.mark.parametrize(("sdk", "label", "key_env"), PROVIDERS)
def test_rate_limit_error_is_recognised(sdk, label, key_env):
    message = describe_sdk_error(
        sdk, _make(sdk, "rate"), provider=label, model="m", key_env=key_env
    )
    assert "Rate limited" in message and label in message


@pytest.mark.parametrize(("sdk", "label", "key_env"), PROVIDERS)
def test_not_found_names_the_model(sdk, label, key_env):
    message = describe_sdk_error(
        sdk, _make(sdk, "notfound"), provider=label, model="some-model", key_env=key_env
    )
    assert "some-model" in message


@pytest.mark.parametrize(("sdk", "label", "key_env"), PROVIDERS)
def test_connection_error_points_at_the_network(sdk, label, key_env):
    message = describe_sdk_error(
        sdk, _make(sdk, "conn"), provider=label, model="m", key_env=key_env
    )
    assert "network" in message.lower()


@pytest.mark.parametrize(("sdk", "label", "key_env"), PROVIDERS)
def test_unknown_exception_still_produces_a_message(sdk, label, key_env):
    message = describe_sdk_error(
        sdk, RuntimeError("something odd"), provider=label, model="m", key_env=key_env
    )
    assert "something odd" in message
    assert label in message


def test_both_providers_get_the_same_shape_of_message():
    """One ladder, so neither provider can drift into worse errors than the other."""
    a = describe_sdk_error(
        anthropic, _make(anthropic, "rate"), provider="X", model="m", key_env="K"
    )
    b = describe_sdk_error(openai, _make(openai, "rate"), provider="X", model="m", key_env="K")
    assert a == b


def test_openai_client_honours_a_self_hosted_base_url():
    client = build_openai_client(
        Settings(openai_api_key=None, openai_base_url="http://localhost:11434/v1")
    )
    assert str(client.base_url).startswith("http://localhost:11434")
    assert client.api_key, "a self-hosted gateway still needs a non-empty key"


def test_openai_client_uses_the_configured_key():
    client = build_openai_client(Settings(openai_api_key="sk-test-123"))
    assert client.api_key == "sk-test-123"
