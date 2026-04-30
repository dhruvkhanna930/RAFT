"""DeepSeek API client (DeepSeek-V3 / deepseek-chat by default).

Cheapest strong model for final evaluation (~$0.27/M input tokens as of
research plan writing — verify current pricing before large runs).
Reads ``DEEPSEEK_API_KEY`` from the environment.

DeepSeek's API is OpenAI-compatible so we use the openai SDK with a custom
base_url.
"""

from __future__ import annotations

import os
from typing import Any


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient:
    """Wrapper around the DeepSeek chat API (OpenAI-compatible).

    Args:
        model: DeepSeek model ID (``"deepseek-chat"`` = DeepSeek-V3).
        api_key: API key (defaults to ``DEEPSEEK_API_KEY`` env var).
        base_url: DeepSeek API base URL.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the DeepSeek chat completions API.

        Args:
            prompt: Full prompt string.
            **kwargs: Override instance defaults (model, temperature, max_tokens).

        Returns:
            Generated text content string.
        """
        from src.llms.openai_client import OpenAIClient
        if self._client is None:
            self._client = OpenAIClient(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._client.generate(prompt, **kwargs)
