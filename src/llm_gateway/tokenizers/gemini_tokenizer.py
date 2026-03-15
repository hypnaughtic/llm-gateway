"""Token counter for Google Gemini models via google-genai SDK."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GeminiTokenizer:
    """Token counter for Google Gemini models.

    Primary: Uses google-genai SDK's count_tokens() API for exact counts.
    Fallback: Calibrated heuristic (chars / 3.5) if SDK unavailable or API call fails.
    """

    HEURISTIC_CHARS_PER_TOKEN = 3.5

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self._model_name = model
        self._client: object | None = None  # google.genai.Client
        self._use_heuristic = False
        self._initialized = False

    def _init_client(self) -> None:
        """Lazy-init: try google-genai SDK client, fall back to heuristic."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from google import genai

            self._client = genai.Client()
            logger.debug("Gemini tokenizer initialized via google-genai SDK")
        except (ImportError, Exception) as exc:
            self._use_heuristic = True
            logger.debug(
                "google-genai SDK not available — using heuristic tokenizer (chars / %.1f): %s",
                self.HEURISTIC_CHARS_PER_TOKEN,
                exc,
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's tokenizer (SDK API call with heuristic fallback)."""
        if not text:
            return 0
        self._init_client()

        if self._use_heuristic:
            return max(1, int(len(text) / self.HEURISTIC_CHARS_PER_TOKEN))

        try:
            from google import genai

            client: genai.Client = self._client  # type: ignore[assignment]
            result = client.models.count_tokens(
                model=self._model_name,
                contents=text,
            )
            total: int = result.total_tokens or 0
            return total
        except Exception as exc:
            logger.warning("Gemini SDK count_tokens failed, falling back to heuristic: %s", exc)
            return max(1, int(len(text) / self.HEURISTIC_CHARS_PER_TOKEN))

    @property
    def name(self) -> str:
        """Human-readable tokenizer name."""
        return "gemini"
