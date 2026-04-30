"""OpenAI API client (gpt-4o-mini by default).

Used for final evaluation tables.  Reads ``OPENAI_API_KEY`` from
the environment.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 5, 10, 20]  # seconds; covers rate-limits and brief network drops


class OpenAIClient:
    """Wrapper around the OpenAI chat completions API.

    Args:
        model: OpenAI model ID.
        api_key: API key (defaults to ``OPENAI_API_KEY`` env var).
        base_url: Override API base URL (for OpenAI-compatible endpoints).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the OpenAI chat completions endpoint.

        Args:
            prompt: Full prompt string.
            **kwargs: Override instance defaults (model, temperature, max_tokens).

        Returns:
            Generated text content string.

        Raises:
            RuntimeError: After all retries are exhausted.
        """
        model = kwargs.get("model", self.model)
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        client = self._get_client()

        last_exc: Exception | None = None
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                logger.warning("OpenAI attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"OpenAI generate failed after retries: {last_exc}") from last_exc
