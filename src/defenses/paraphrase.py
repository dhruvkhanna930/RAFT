"""Paraphrase defenses — LLM-based rewriting at passage and query level.

Two distinct defenses in this module:

- :class:`ParaphraseDefense` — rewrites *retrieved passages* before they are
  fed to the generator.  Tests whether paraphrasing removes the adversarial
  signal.  Hypothesis: invisible chars survive prompts that do not explicitly
  strip non-ASCII characters.

- :class:`QueryParaphraseDefense` — rewrites the *user query* before
  retrieval.  Rationale: if the query is rephrased the embedding changes
  slightly, potentially shifting which passages rank in top-k.  Hypothesis:
  minimal effect against PoisonedRAG (passage text unchanged) but some
  reduction against RAG-Pull (query embedding drift lowers cosine similarity
  with optimised adversarial passage).
"""

from __future__ import annotations

from typing import Any

from src.defenses.base import DefenseBase


class ParaphraseDefense(DefenseBase):
    """Paraphrase retrieved passages using an LLM before generation.

    Satisfies :class:`DefenseBase`: ``apply`` accepts a single string or a
    list.

    Args:
        llm: LLM client with a ``generate(prompt, **kwargs) -> str`` method.
        prompt_template: Template with a ``{passage}`` placeholder.
    """

    DEFAULT_PROMPT = (
        "Paraphrase the following passage in your own words, "
        "preserving the factual content:\n\n{passage}"
    )

    def __init__(
        self,
        llm: Any,
        prompt_template: str | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    # ── DefenseBase interface ─────────────────────────────────────────────────

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Paraphrase a single passage or a list of passages.

        Args:
            text_or_passages: Passage string or list of passage strings.

        Returns:
            Paraphrased string or list of paraphrased strings.
        """
        if isinstance(text_or_passages, str):
            return self._paraphrase(text_or_passages)
        return [self._paraphrase(p) for p in text_or_passages]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _paraphrase(self, passage: str) -> str:
        """Paraphrase a single passage string.

        Args:
            passage: Input passage text.

        Returns:
            Paraphrased passage text.
        """
        prompt = self.prompt_template.format(passage=passage)
        return self.llm.generate(prompt)


class QueryParaphraseDefense(DefenseBase):
    """Rewrite the user query with an LLM before retrieval.

    Satisfies :class:`DefenseBase`: ``apply`` accepts a single query string
    or a list of query strings and returns the same type.

    The rewritten query has a slightly different embedding, which may reduce
    cosine similarity between the query and adversarial passages that were
    optimised against the *original* query embedding (RAG-Pull / Hybrid).
    Against PoisonedRAG the effect is expected to be minimal because those
    passages rely on semantic content, not embedding proximity.

    Args:
        llm: LLM client with a ``generate(prompt, **kwargs) -> str`` method.
        prompt_template: Template with a ``{query}`` placeholder.
        max_tokens: Token budget for the rewrite call.
    """

    DEFAULT_PROMPT = (
        "Rephrase the following question in a different way while keeping "
        "the same meaning. Output only the rephrased question, nothing else."
        "\n\nQuestion: {query}\nRephrased question:"
    )

    def __init__(
        self,
        llm: Any,
        prompt_template: str | None = None,
        max_tokens: int = 60,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.max_tokens = max_tokens

    # ── DefenseBase interface ─────────────────────────────────────────────────

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Rewrite a single query or a list of queries.

        Args:
            text_or_passages: A query string or list of query strings.

        Returns:
            Rewritten query string or list of rewritten query strings.
        """
        if isinstance(text_or_passages, str):
            return self._rewrite(text_or_passages)
        return [self._rewrite(q) for q in text_or_passages]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rewrite(self, query: str) -> str:
        """Rewrite a single query string.

        Args:
            query: Original user query.

        Returns:
            Rephrased query string (first non-empty line of LLM output).
        """
        prompt = self.prompt_template.format(query=query)
        raw = self.llm.generate(prompt, temperature=0.3, max_tokens=self.max_tokens)
        # Take first non-empty line; strip leading/trailing whitespace.
        for line in raw.splitlines():
            line = line.strip()
            if line:
                return line
        return query  # fallback: return original if LLM output is empty
