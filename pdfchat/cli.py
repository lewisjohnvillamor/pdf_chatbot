"""Small operational CLI.

``python -m pdfchat.cli hash-password`` generates the APP_PASSWORD_HASH value
for a deployment; ``check-config`` validates the environment before a rollout.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from dotenv import load_dotenv

from .config import load_settings
from .errors import PdfChatError
from .security import hash_password


def _hash_password(args: argparse.Namespace) -> int:
    password = args.password or getpass.getpass("Password: ")
    if not password:
        print("A password is required.", file=sys.stderr)
        return 1
    if not args.password and password != getpass.getpass("Confirm: "):
        print("Passwords do not match.", file=sys.stderr)
        return 1
    print(f"APP_PASSWORD_HASH={hash_password(password)}")
    return 0


def _check_config(_args: argparse.Namespace) -> int:
    load_dotenv()
    try:
        settings = load_settings()
    except PdfChatError as exc:
        print(f"Invalid configuration: {exc.user_message}", file=sys.stderr)
        return 1

    missing = settings.missing_credentials()
    print(f"chat:       {settings.chat_provider}/{settings.chat_model}")
    print(f"embeddings: {settings.embedding_provider}/{settings.embedding_model}")
    print(f"store:      {settings.vector_store}")
    print(f"auth:       {'enabled' if settings.auth_enabled else 'DISABLED'}")
    if missing:
        print(f"missing credentials: {', '.join(missing)}", file=sys.stderr)
        return 1
    if not settings.auth_enabled:
        print(
            "warning: APP_PASSWORD_HASH is unset — do not expose this instance publicly.",
            file=sys.stderr,
        )
    print("configuration OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdfchat", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hasher = subparsers.add_parser("hash-password", help="generate APP_PASSWORD_HASH")
    hasher.add_argument("--password", help="read from a prompt when omitted (preferred)")
    hasher.set_defaults(func=_hash_password)

    checker = subparsers.add_parser("check-config", help="validate the environment")
    checker.set_defaults(func=_check_config)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
