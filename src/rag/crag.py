"""CRAG — Corrective RAG (Yan et al., 2024).

A lightweight retrieval evaluator scores retrieved passages and routes each
to one of three actions: Correct (use as-is), Incorrect (discard), or
Ambiguous (refine and use).  Web-search fallback is disabled during eval.

**Deviation from original:** The upstream evaluator is a fine-tuned T5
cross-encoder that classifies passages as correct/incorrect/ambiguous.
When ``evaluator=None`` (default), we use the retrieval cosine similarity
scores returned by the retriever as the relevance proxy — they are already
in [0, 1] for normalised-embedding retrievers like Contriever.  This
preserves the three-way routing logic without requiring a separate model.

Upstream repo: https://github.com/HuskyInSalt/CRAG
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from src.rag.base import GenerationResult, RagBase, RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalAction(str, Enum):
    """CRAG three-way routing decision."""

    CORRECT = "correct"       # high relevance — use passage directly
    INCORRECT = "incorrect"   # low relevance  — discard
    AMBIGUOUS = "ambiguous"   # medium relevance — refine then use


_REFINE_PROMPT = """\
Given the following passage and question, rewrite the question to be more \
specific so that it can be answered by the passage. Return only the rewritten \
question, nothing else.

Passage: {passage}
Original question: {question}
Rewritten question:"""

_GENERATION_PROMPT = """\
Answer the question based on the given passages. \
Only give me the answer and do not output any other words.

The following are given passages.
{passages}

Answer the question based on the given passages. \
Only give me the answer and do not output any other words.
Question: {question}
Answer:"""


class CRAG(RagBase):
    """Corrective RAG pipeline with a lightweight relevance evaluator.

    Passages above ``upper_threshold`` are used directly; below
    ``lower_threshold`` are discarded; in between trigger query refinement
    via the LLM.

    Args:
        retriever: Base retriever for initial passage lookup.
        llm: LLM for answer generation and query refinement.
        evaluator: Optional external relevance evaluator callable.
            Signature: ``evaluator(query, passage) -> float`` returning a
            score in [0, 1].  When ``None``, retrieval cosine similarity
            scores are used instead.
        top_k: Number of passages to retrieve initially.
        upper_threshold: Relevance threshold for CORRECT action.
        lower_threshold: Relevance threshold for INCORRECT action.
        use_web_fallback: Set False during evaluation (no internet).
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        evaluator: Any = None,
        top_k: int = 5,
        upper_threshold: float = 0.8,
        lower_threshold: float = 0.4,
        use_web_fallback: bool = False,
    ) -> None:
        super().__init__(retriever, llm, top_k)
        self.evaluator = evaluator
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.use_web_fallback = use_web_fallback

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        """Retrieve and evaluate passages; apply CRAG routing.

        1. Retrieve *k* passages via the base retriever.
        2. Score each passage (external evaluator or retrieval score).
        3. Route each via ``_route(score)`` — CORRECT / AMBIGUOUS / INCORRECT.
        4. Keep CORRECT passages as-is.
        5. For AMBIGUOUS passages: refine the query and re-retrieve one passage.
        6. Discard INCORRECT passages.

        Args:
            query: User question.
            k: Initial retrieval count (before filtering).

        Returns:
            ``RetrievalResult`` after CRAG filtering with routing metadata.
        """
        passages, scores = self.retriever.retrieve(query, k=k)

        # Score with external evaluator if provided.
        if self.evaluator is not None:
            scores = [
                float(self.evaluator(query, p)) for p in passages
            ]

        actions: list[RetrievalAction] = [self._route(s) for s in scores]

        kept_passages: list[str] = []
        kept_scores: list[float] = []
        routing_log: list[dict[str, Any]] = []

        for passage, score, action in zip(passages, scores, actions):
            routing_log.append({
                "passage_preview": passage[:80],
                "score": score,
                "action": action.value,
            })

            if action == RetrievalAction.CORRECT:
                kept_passages.append(passage)
                kept_scores.append(score)

            elif action == RetrievalAction.AMBIGUOUS:
                refined_query = self._refine_query(query, passage)
                try:
                    ref_passages, ref_scores = self.retriever.retrieve(
                        refined_query, k=1
                    )
                    if ref_passages:
                        kept_passages.append(ref_passages[0])
                        kept_scores.append(ref_scores[0])
                except Exception:
                    logger.debug(
                        "CRAG: refined retrieval failed for %r — skipping",
                        refined_query[:60],
                    )
            # INCORRECT → discard silently.

        # Fallback: if all passages were discarded, use the top-1 original.
        if not kept_passages:
            logger.debug(
                "CRAG: all passages discarded for %r — falling back to top-1",
                query[:60],
            )
            kept_passages = [passages[0]]
            kept_scores = [scores[0]]

        return RetrievalResult(
            passages=kept_passages,
            scores=kept_scores,
            metadata={"routing": routing_log},
        )

    def generate(self, query: str, retrieved: RetrievalResult) -> GenerationResult:
        """Generate answer from CRAG-filtered passages.

        Uses the same PoisonedRAG-style prompt template as VanillaRAG to
        keep ASR measurement consistent.

        Args:
            query: User question.
            retrieved: CRAG-filtered ``RetrievalResult``.

        Returns:
            ``GenerationResult`` with routing metadata.
        """
        passages_block = "\n\n".join(
            f"Passage {i + 1}: {p}"
            for i, p in enumerate(retrieved.passages)
        )
        prompt = _GENERATION_PROMPT.format(
            passages=passages_block,
            question=query,
        )
        answer = self.llm.generate(prompt)
        return GenerationResult(
            answer=answer,
            retrieved=retrieved,
            metadata={"routing": retrieved.metadata.get("routing", [])},
        )

    def _route(self, score: float) -> RetrievalAction:
        """Classify a passage relevance score into a CRAG action.

        Args:
            score: Relevance score (cosine similarity or evaluator output).

        Returns:
            ``RetrievalAction`` enum value.
        """
        if score >= self.upper_threshold:
            return RetrievalAction.CORRECT
        if score <= self.lower_threshold:
            return RetrievalAction.INCORRECT
        return RetrievalAction.AMBIGUOUS

    def _refine_query(self, query: str, passage: str) -> str:
        """Use the LLM to rewrite the query so it better targets the passage.

        Args:
            query: Original user question.
            passage: Ambiguous passage text.

        Returns:
            Refined query string.
        """
        prompt = _REFINE_PROMPT.format(question=query, passage=passage)
        refined = self.llm.generate(prompt).strip()
        # Sanity: if the LLM returns empty or very short, fall back.
        if len(refined) < 5:
            return query
        return refined
