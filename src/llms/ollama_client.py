"""Ollama local LLM client.

Wraps the Ollama REST API (``http://localhost:11434``) for local inference.
Used for all development and iteration runs before moving to paid APIs.

Supported models (pull with ``ollama pull <model_id>``):
- llama3.1:8b  (default)
- qwen2.5:7b
- mistral:7b
- phi3:mini
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Timeout for a single generate call — long enough for cold-start model loading.
_DEFAULT_TIMEOUT = 300  # seconds


class OllamaClient:
    """Thin wrapper around the Ollama /api/generate REST endpoint.

    All parameters can be overridden per-call via keyword arguments to
    :meth:`generate`.

    Args:
        model: Ollama model tag (e.g. ``"qwen2.5:7b"``).
        base_url: Ollama server URL.
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum new tokens to generate (``num_predict``).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 200,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send *prompt* to Ollama and return the generated text.

        Args:
            prompt: Full prompt string (context + question already formatted).
            **kwargs: Per-call overrides — ``model``, ``temperature``,
                ``max_tokens``.

        Returns:
            Generated text string (leading/trailing whitespace stripped).

        Raises:
            RuntimeError: If Ollama is unreachable, times out, or returns an
                HTTP error.
        """
        model = kwargs.get("model", self.model)
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        logger.debug("POST /api/generate  model=%s  tokens=%d", model, max_tokens)
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}. "
                "Start it with: ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s "
                f"(model={model}). Try a smaller model or increase timeout."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            body = exc.response.text[:300] if exc.response is not None else ""
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        return resp.json()["response"].strip()

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Returns:
            ``True`` if the server responds to GET /api/tags with HTTP 200.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Return the names of all locally-pulled models.

        Returns:
            List of model name strings (e.g. ``["qwen2.5:7b", "phi3:latest"]``).
            Returns an empty list if the server is unreachable.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.RequestException:
            return []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, base_url={self.base_url!r})"
        )
