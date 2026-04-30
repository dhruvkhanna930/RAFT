"""LLM client wrappers for local (Ollama) and API-based models.

All clients expose a common ``generate(prompt, **kwargs)`` interface.
"""

from src.llms.ollama_client import OllamaClient
from src.llms.openai_client import OpenAIClient
from src.llms.anthropic_client import AnthropicClient
from src.llms.deepseek_client import DeepSeekClient
from src.llms.groq_client import GroqClient

__all__ = ["OllamaClient", "OpenAIClient", "AnthropicClient", "DeepSeekClient", "GroqClient"]
