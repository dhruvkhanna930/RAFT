"""Anthropic API client (Claude Haiku 4.5 by default).

Used for final evaluation tables.  Reads ``ANTHROPIC_API_KEY`` from
the environment.
"""

from __future__ import annotations

import os
from typing import Any


class AnthropicClient:
    """Wrapper around the Anthropic messages API.

    Args:
        model: Anthropic model ID.
        api_key: API key (defaults to ``ANTHROPIC_API_KEY`` env var).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the Anthropic messages API.

        Args:
            prompt: Full prompt string (treated as a user message).
            **kwargs: Override instance defaults.

        Returns:
            Generated text content string.
        """
        # TODO: import anthropic; call client.messages.create(...)
        #       extract content[0].text
        raise NotImplementedError
