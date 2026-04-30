"""Groq API client — fast free inference for LLaMA / Mixtral models.

Groq's API is OpenAI-compatible; this thin wrapper points the OpenAI SDK
at api.groq.com.  Reads ``GROQ_API_KEY`` from the environment.

Free-tier limits (as of 2025):
  - llama-3.1-8b-instant: 30 req/min, 14 400 req/day
  - llama-3.3-70b-versatile: 30 req/min, 14 400 req/day

Get a free key at https://console.groq.com

Usage::

    export GROQ_API_KEY=gsk_...
    llm = GroqClient(model="llama-3.1-8b-instant")
    answer = llm.generate("What is the capital of France?")
"""

from __future__ import annotations

import os
from typing import Any

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient:
    """Wrapper around the Groq chat completions API (OpenAI-compatible).

    Args:
        model: Groq model ID.
        api_key: API key (defaults to ``GROQ_API_KEY`` env var).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._delegate: Any = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the Groq chat completions endpoint.

        Args:
            prompt: Full prompt string.
            **kwargs: Override instance defaults (model, temperature, max_tokens).

        Returns:
            Generated text content string.

        Raises:
            RuntimeError: If the API key is missing or after all retries fail.
        """
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        if self._delegate is None:
            from src.llms.openai_client import OpenAIClient
            self._delegate = OpenAIClient(
                model=self.model,
                api_key=self.api_key,
                base_url=GROQ_BASE_URL,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._delegate.generate(prompt, **kwargs)

    def is_available(self) -> bool:
        """Return True if GROQ_API_KEY is set and a test call succeeds."""
        if not self.api_key:
            return False
        try:
            result = self.generate("ping", max_tokens=3)
            return isinstance(result, str)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
