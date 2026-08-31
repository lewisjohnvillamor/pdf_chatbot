"""Access control and abuse limits for a self-hosted deployment.

Two concerns, both mandatory once the app is reachable beyond localhost:
a password gate so an exposed instance is not an open API-key proxy, and a
per-session rate limit so one user cannot exhaust a shared budget.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: PBKDF2 cost. High enough to make offline cracking expensive, low enough
#: that a login stays imperceptible.
_ITERATIONS = 240_000
_ALGORITHM = "sha256"


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Hash a password into the ``pbkdf2_sha256$iterations$salt$hash`` format."""
    if not password:
        raise ValueError("password must not be empty")
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(_ALGORITHM, password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_{_ALGORITHM}${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against a stored hash."""
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$")
        if not algorithm.startswith("pbkdf2_"):
            return False
        candidate = hashlib.pbkdf2_hmac(
            algorithm.removeprefix("pbkdf2_"),
            password.encode(),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        logger.warning("password_hash_malformed")
        return False
    return hmac.compare_digest(candidate.hex(), digest_hex)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass(slots=True)
class RateLimiter:
    """Sliding-window request limiter, keyed by session.

    In-process state is the right scope here: each Streamlit session gets its
    own limiter, and a shared deployment bounds concurrency at the reverse
    proxy. It exists to stop one runaway session, not to be a global quota.
    """

    max_events: int
    window_seconds: float = 3600.0
    _events: list[float] = field(default_factory=list)

    def allow(self) -> bool:
        """Record an event and report whether it was within the limit."""
        if self.max_events <= 0:
            return True
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._events = [t for t in self._events if t > cutoff]
        if len(self._events) >= self.max_events:
            logger.warning("rate_limit_exceeded", extra={"limit": self.max_events})
            return False
        self._events.append(now)
        return True

    @property
    def remaining(self) -> int:
        if self.max_events <= 0:
            return -1
        cutoff = time.monotonic() - self.window_seconds
        return max(0, self.max_events - len([t for t in self._events if t > cutoff]))

    def retry_after_seconds(self) -> int:
        """Seconds until the oldest event leaves the window."""
        if not self._events:
            return 0
        return max(0, int(self.window_seconds - (time.monotonic() - self._events[0])) + 1)
