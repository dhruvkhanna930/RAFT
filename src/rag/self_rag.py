"""Self-RAG variant (Asai et al., ICLR 2024).

The LLM generates special reflection tokens to decide whether to retrieve,
how many times, and whether retrieved passages are relevant / supported.

**Deviation from original:** The upstream model is a fine-tuned Llama-2 7B with
custom vocabulary tokens ([Retrieve], [Relevant], [Irrelevant], [Supported],
[Partially Supported], [No Support], [Utility]).  Fine-tuning a 7B model is
outside scope, so we approximate the reflection-token mechanism via prompting:
for each passage, the LLM is asked whether the passage is relevant to the query
and only passages judged "yes" are kept as context.  This preserves the *spirit*
of Self-RAG (per-passage relevance gating) without requiring a custom model.

Upstream repo: https://github.com/AkariAsai/self-rag
Key dependency: transformers (or vllm for faster inference).
Model: selfrag/selfrag_llama2_7b on HuggingFace.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.base import GenerationResult, RagBase, RetrievalResult

logger = logging.getLogger(__name__)

_RELEVANCE_PROMPT = """\
Is the following passage relevant to answering the question?
Question: {question}
Passage: {passage}
Answer with exactly "Yes" or "No"."""

_GENERATION_PROMPT = """\
Answer the question based on the given passages. \
Only give me the answer and do not output any other words.

The following are given passages.
{passages}

Answer the question based on the given passages. \
Only give me the answer and do not output any other words.
Question: {question}
Answer:"""


class SelfRAG(RagBase):
    """Self-RAG wrapper implementing adaptive retrieve-then-reflect.

    For each retrieved passage, the LLM judges relevance via a yes/no prompt.
    Only passages deemed relevant are used as context for the final generation.
    If no passages pass the relevance check, all are used as a fallback
    (graceful degradation).

    Args:
        retriever: Retriever used for passage lookup.
        llm: LLM client (must implement ``generate(prompt) -> str``).
        top_k: Passages fetched per retrieval trigger.
        beam_width: Number of beams for tree-structured decoding.
            Kept for constructor compatibility; not used in the
            prompt-based approximation.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        top_k: int = 5,
        beam_width: int = 2,
    ) -> None:
        super().__init__(retriever, llm, top_k)
        self.beam_width = beam_width

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        """Retrieve passages via the base retriever.

        Args:
            query: User question.
            k: Number of passages to fetch.

        Returns:
            ``RetrievalResult`` with passages and scores.
        """
        passages, scores = self.retriever.retrieve(query, k=k)
        return RetrievalResult(passages=passages, scores=scores)

    def generate(self, query: str, retrieved: RetrievalResult) -> GenerationResult:
        """Run prompt-based Self-RAG: judge relevance per passage, then generate.

        For each passage the LLM is asked "Is this passage relevant?"  Only
        passages judged relevant are kept as context.  Falls back to all
        passages if none pass the relevance check.

        Args:
            query: User question.
            retrieved: Initial retrieval result.

        Returns:
            ``GenerationResult`` with answer and per-passage relevance
            verdicts in ``metadata["relevance"]``.
        """
        relevance: list[bool] = []
        for passage in retrieved.passages:
            prompt = _RELEVANCE_PROMPT.format(question=query, passage=passage)
            verdict = self.llm.generate(prompt).strip().lower()
            is_relevant = verdict.startswith("yes")
            relevance.append(is_relevant)

        relevant_passages = [
            p for p, r in zip(retrieved.passages, relevance) if r
        ]
        relevant_scores = [
            s for s, r in zip(retrieved.scores, relevance) if r
        ]

        # Graceful degradation: if nothing is relevant, use all passages.
        if not relevant_passages:
            logger.debug(
                "Self-RAG: no passages judged relevant for %r — using all %d",
                query[:60],
                len(retrieved.passages),
            )
            relevant_passages = retrieved.passages
            relevant_scores = retrieved.scores

        passages_block = "\n\n".join(
            f"Passage {i + 1}: {p}"
            for i, p in enumerate(relevant_passages)
        )
        gen_prompt = _GENERATION_PROMPT.format(
            passages=passages_block,
            question=query,
        )
        answer = self.llm.generate(gen_prompt)

        filtered = RetrievalResult(
            passages=relevant_passages,
            scores=relevant_scores,
            metadata={"relevance": relevance},
        )
        return GenerationResult(
            answer=answer,
            retrieved=filtered,
            metadata={"relevance": relevance},
        )
