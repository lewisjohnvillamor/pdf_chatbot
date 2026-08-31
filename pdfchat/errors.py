"""Exception hierarchy for the PDF chatbot.

Every error surfaced to the UI derives from :class:`PdfChatError` so the
Streamlit layer can show a clean message instead of a traceback, while
unexpected exceptions keep bubbling up to the logs.
"""

from __future__ import annotations


class PdfChatError(Exception):
    """Base class for all errors this application raises deliberately."""

    #: Message shown to end users. Subclasses may override for friendlier text.
    user_message = "Something went wrong while processing your request."

    def __init__(self, message: str = "", *, user_message: str | None = None):
        super().__init__(message or self.user_message)
        if user_message is not None:
            self.user_message = user_message
        elif message:
            self.user_message = message


class ConfigError(PdfChatError):
    """Raised when the runtime configuration is missing or inconsistent."""


class IngestionError(PdfChatError):
    """Raised when a PDF cannot be read, is too large, or has no text layer."""


class EmbeddingError(PdfChatError):
    """Raised when an embedding backend fails or is unavailable."""


class ProviderError(PdfChatError):
    """Raised when the chat provider fails after retries."""


class RetrievalError(PdfChatError):
    """Raised when the index cannot answer a search request."""
