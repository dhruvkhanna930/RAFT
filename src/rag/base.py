"""Abstract base class for all RAG pipeline variants.

Defines the two-method contract that every concrete RAG must satisfy.
Experiment scripts call only ``retrieve`` and ``generate`` (or the
``answer`` convenience wrapper), making it trivial to swap variants.

Corpus loading is separated from retrieval: call ``load_corpus(passages)``
once to index a corpus, then call ``retrieve(query, k)`` for each query.
This keeps the per-query hot-path free of corpus-size assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Output of a single ``retrieve`` call.

    Args:
        passages: Retrieved passage strings in descending-score order.
        scores: Corresponding similarity / relevance scores.
        metadata: Retriever-specific extras (doc IDs, cluster labels, …).
    """

    passages: list[str]
    scores: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output of a single ``generate`` call.

    Args:
        answer: The LLM's answer string.
        retrieved: The :class:`RetrievalResult` used as context.
        metadata: RAG-variant-specific extras (self-critique tokens,
            trust scores, per-passage isolated generations, …).
    """

    answer: str
    retrieved: RetrievalResult
    metadata: dict[str, Any] = field(default_factory=dict)


class RagBase(ABC):
    """Abstract base class for RAG pipeline variants.

    Contract
    --------
    1. Call :meth:`load_corpus` once to index the passage corpus.
    2. Call :meth:`retrieve` for each query to get top-k passages.
    3. Call :meth:`generate` with the retrieval result to get an answer.
    4. Or call :meth:`answer` which chains steps 2–3 for you.

    The corpus is held by ``self.retriever``; neither ``retrieve`` nor
    ``generate`` accept a corpus argument, so no scale assumptions leak
    into the calling interface.

    Args:
        retriever: An instantiated retriever (must implement
            ``build_index(corpus)`` and ``retrieve(query, k)``).
        llm: An instantiated LLM client (must implement ``generate(prompt)``).
        top_k: Default number of passages to retrieve when *k* is not
            supplied to :meth:`retrieve` or :meth:`answer`.
    """

    def __init__(self, retriever: Any, llm: Any, top_k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    # ── Corpus loading ────────────────────────────────────────────────────────

    def load_corpus(self, corpus: list[str]) -> None:
        """Index *corpus* into ``self.retriever``.

        Must be called before any ``retrieve`` / ``answer`` call.
        Delegates to ``self.retriever.build_index``; subclasses may override
        to add variant-specific preprocessing (e.g. TrustRAG pre-embeds).

        Args:
            corpus: List of passage strings to index.
        """
        self.retriever.build_index(corpus)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def retrieve(self, query: str, k: int) -> RetrievalResult:
        """Retrieve the top-*k* passages most relevant to *query*.

        The corpus must already be indexed via :meth:`load_corpus`.

        Args:
            query: User question string.
            k: Number of passages to retrieve.

        Returns:
            :class:`RetrievalResult` with passages ranked by score.
        """

    @abstractmethod
    def generate(self, query: str, retrieved: RetrievalResult) -> GenerationResult:
        """Generate an answer given *query* and *retrieved* passages.

        Args:
            query: User question string.
            retrieved: Output of :meth:`retrieve`.

        Returns:
            :class:`GenerationResult` with the LLM's answer.
        """

    # ── Convenience ───────────────────────────────────────────────────────────

    def answer(self, query: str, k: int | None = None) -> GenerationResult:
        """Retrieve then generate in one call.

        Args:
            query: User question string.
            k: Passages to retrieve.  Defaults to ``self.top_k``.

        Returns:
            Final :class:`GenerationResult`.
        """
        retrieved = self.retrieve(query, k if k is not None else self.top_k)
        return self.generate(query, retrieved)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"


# Backwards-compatibility alias — remove once all imports are updated.
BaseRAG = RagBase
