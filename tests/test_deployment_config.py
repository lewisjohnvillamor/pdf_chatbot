"""Guards on the shipped deployment configuration.

These are not unit tests of Python behaviour — they are tests of the files a
user copy-pastes into production. A published default database password is a
real vulnerability in a self-hosted app, and it regresses silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"

#: Matches a URI carrying an inline lowercase user/password pair. Uppercase
#: placeholders are allowed, so documented examples do not trip it. The literal
#: pattern is not spelled out here: this file is itself scanned, and an example
#: in a comment would make the check fail on its own documentation.
CREDENTIAL_URI = re.compile(r"://(?!USER:PASSWORD)[a-z0-9_]+:[a-z0-9_]+@")


def tracked_text_files() -> list[Path]:
    import subprocess

    output = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    keep = {".py", ".yml", ".yaml", ".toml", ".md", ".sql", ".json", ".example", ".txt"}
    return [
        ROOT / name
        for name in output
        if (ROOT / name).suffix in keep or name.endswith(".env.example")
    ]


def test_compose_has_no_default_database_password():
    """Compose must fail closed, not fall back to a password published here."""
    text = COMPOSE.read_text()
    assert "POSTGRES_PASSWORD:-" not in text, (
        "docker-compose.yml provides a default POSTGRES_PASSWORD. A deployment "
        "that forgets to set one would run with a password anyone can read in "
        "this repository. Use ${POSTGRES_PASSWORD:?message} so compose refuses "
        "to start instead."
    )
    assert "POSTGRES_PASSWORD:?" in text


def test_compose_error_message_tells_the_user_what_to_do():
    text = COMPOSE.read_text()
    match = re.search(r"POSTGRES_PASSWORD:\?([^}\"]+)", text)
    assert match, "the required-variable guard must carry a message"
    assert "openssl rand" in match.group(1), "the message should show how to generate one"


def _is_credential_key(key: str) -> bool:
    """Names that hold a secret — not merely names containing 'token'.

    MAX_OUTPUT_TOKENS is a budget, not a credential; matching it would make
    this check noisy enough that someone would delete it.
    """
    key = key.upper()
    return "PASSWORD" in key or "SECRET" in key or key.endswith("API_KEY") or key.endswith("_TOKEN")


def test_env_example_ships_no_populated_secret():
    """Every credential field in .env.example must be blank."""
    for line in ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if _is_credential_key(key):
            assert not value.strip(), f"{key} in .env.example ships a value: {value!r}"


def test_credential_key_detection_is_precise():
    assert _is_credential_key("ANTHROPIC_API_KEY")
    assert _is_credential_key("POSTGRES_PASSWORD")
    assert _is_credential_key("APP_PASSWORD_HASH")
    assert not _is_credential_key("MAX_OUTPUT_TOKENS")
    assert not _is_credential_key("RATE_LIMIT_QUESTIONS_PER_HOUR")


def test_no_tracked_file_contains_an_inline_credential_uri():
    """A user:password@host URI is a secret-scanner finding and a bad example."""
    offenders: list[str] = []
    for path in tracked_text_files():
        if path.resolve() == Path(__file__).resolve():
            continue  # this file defines the pattern; scanning it is circular
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if CREDENTIAL_URI.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "inline credentials found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("required", ["APP_PASSWORD_HASH", "POSTGRES_PASSWORD", "RERANKER"])
def test_env_example_documents_every_security_relevant_setting(required):
    assert required in ENV_EXAMPLE.read_text()
