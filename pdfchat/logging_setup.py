"""Structured JSON logging.

Production deployments ship stdout to a log aggregator, so records are emitted
as single-line JSON with a stable field set. Anything passed via ``extra=`` is
merged into the payload, which is how the ingestion and retrieval paths attach
document ids, latencies and token counts.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as one-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger exactly once."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        if isinstance(getattr(handler, "formatter", None), JsonFormatter):
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    # These libraries log a request line per call; keep them out of the way.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def log_duration(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log ``event`` with its wall-clock duration, plus anything the body adds."""
    started = time.perf_counter()
    extra: dict[str, Any] = dict(fields)
    try:
        yield extra
    except Exception:
        extra["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(f"{event}.failed", extra=extra)
        raise
    extra["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    logger.info(event, extra=extra)
