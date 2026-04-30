"""Hybrid attack: semantic injection + Unicode retrieval trigger.

**Primary research contribution.**

Combines two complementary attack vectors into a single adversarial passage:

Stage 1 — **PoisonedRAG** (semantic payload):
    Craft a passage whose *text* makes the LLM reproduce the target answer
    when it is retrieved.  The generation condition is verified per passage.

Stage 2 — **RAG-Pull** (Unicode trigger):
    Apply invisible-character insertions (zero-width, tag chars, …) to the
    Stage-1 passage using Differential Evolution to maximise its cosine
    similarity to the target query.  This boosts retrieval rank without any
    visible text change.

Key properties:
- ``strip_invisible(hybrid_passage) == semantic_passage`` for all outputs —
  the hybrid passage is *visually identical* to its PoisonedRAG counterpart.
- The generation condition is inherited from Stage 1 (same visible text).
- The retrieval condition is improved by Stage 2 (invisible trigger chars).

Reference: this project; builds on PoisonedRAG + RAG-Pull.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.attacks.base import AttackBase, AttackConfig
from src.attacks.poisoned_rag import PoisonedRAGAttack
from src.attacks.rag_pull import RAGPullAttack
from src.attacks.unicode_chars import UnicodeInventory

logger = logging.getLogger(__name__)


class HybridAttack(AttackBase):
    """Two-stage attack: semantic construction + Unicode embedding boost.

    **Stage 1** (PoisonedRAG): craft a passage whose *text* elicits the
    target answer from the LLM when retrieved.

    **Stage 2** (RAG-Pull / DE): perturb that passage with invisible Unicode
    chars so the retriever ranks it in top-k for the target query.

    Args:
        config: Shared ``AttackConfig``.
        semantic_cfg: Kwargs forwarded to ``PoisonedRAGAttack.__init__``.
            Must include ``"llm"`` (required by PoisonedRAGAttack).
        unicode_cfg: Kwargs forwarded to ``RAGPullAttack.__init__``.
            Should include ``"retriever"`` for DE embedding calls.
        trigger_location: Where to concentrate Unicode chars —
            ``"prefix"`` (first 50 chars), ``"suffix"`` (last 50 chars),
            or ``"interleaved"`` (whitespace positions across the passage).
    """

    def __init__(
        self,
        config: AttackConfig,
        semantic_cfg: dict[str, Any] | None = None,
        unicode_cfg: dict[str, Any] | None = None,
        trigger_location: str = "prefix",
    ) -> None:
        super().__init__(config)
        self.trigger_location = trigger_location
        # keyword_seed=True + prepend_question=False: the LLM prompt includes
        # question keywords so the generated text is topically aligned with the
        # query (good semantic similarity → reliable retrieval).  The question
        # is NOT prepended as a raw prefix, so the passage reads as natural
        # encyclopedic prose — low perplexity — and survives PPL-based filters.
        # Unicode chars (Stage-2) provide an additional retrieval boost on top.
        sem_cfg = dict(semantic_cfg or {})
        sem_cfg.setdefault("keyword_seed", True)
        sem_cfg.setdefault("prepend_question", False)
        self._semantic = PoisonedRAGAttack(config, **sem_cfg)
        self._unicode = RAGPullAttack(config, **(unicode_cfg or {}))

    # ── Public API ─────────────────────────────────────────────────────────────

    def craft_malicious_passages(
        self,
        target_question: str,
        target_answer: str,
        n: int,
    ) -> list[str]:
        """Run Stage 1 then Stage 2 to produce *n* hybrid adversarial passages.

        The returned passages are visually identical to the Stage-1 semantic
        payloads — calling ``strip_invisible(hybrid)`` recovers the exact
        PoisonedRAG text.

        Args:
            target_question: Target question.
            target_answer: Desired LLM answer.
            n: Number of passages to craft.

        Returns:
            List of *n* passage strings with both semantic and Unicode
            perturbation applied.

        Raises:
            RuntimeError: If ``_unicode.retriever`` is not set.
        """
        if self._unicode.retriever is None:
            raise RuntimeError(
                "Set HybridAttack._unicode.retriever before calling "
                "craft_malicious_passages()."
            )

        # ── Stage 1: semantic payload (satisfies generation condition) ─────────
        logger.info(
            "Hybrid  Stage-1 PoisonedRAG  question=%r  n=%d",
            target_question[:60],
            n,
        )
        semantic_texts = self._semantic.craft_malicious_passages(
            target_question, target_answer, n
        )

        # Store for downstream comparison (e.g. in experiment scripts).
        self._last_semantic_passages: list[str] = list(semantic_texts)

        # ── Stage 2: Unicode perturbation (boosts retrieval condition) ─────────
        # Run DE *once* on the first semantic passage to find the optimal
        # insertion set, then apply those same insertions to all n passages.
        # This gives O(1) DE cost regardless of n — the same speedup as
        # RAGPullAttack which also reuses one perturbation across n copies.
        # Positions are in the trigger region (prefix by default, ≤50 chars),
        # which is always within bounds for any LLM-generated passage.
        query_emb: np.ndarray = self._unicode.retriever.encode_query(target_question)

        first = semantic_texts[0]
        start, end = self._locate_trigger_region(first)
        valid_positions = self._trigger_positions(first, start, end)

        _, shared_insertions = self._unicode._optimize_with_insertions(
            first, query_emb, valid_positions
        )
        logger.debug(
            "Hybrid  Stage-2 DE done  shared %d insertions → applying to all %d passages",
            len(shared_insertions),
            n,
        )

        hybrid_texts: list[str] = []
        for idx, sem_text in enumerate(semantic_texts):
            hybrid = self._unicode._apply_perturbation(sem_text, shared_insertions)
            hybrid_texts.append(hybrid)
            logger.debug(
                "Hybrid  passage %d/%d  sem_len=%d  hybrid_len=%d  "
                "invisible_chars=%d",
                idx + 1,
                n,
                len(sem_text),
                len(hybrid),
                len(hybrid) - len(sem_text),
            )

        return hybrid_texts

    def boost_passages(
        self,
        target_question: str,
        semantic_passages: list[str],
    ) -> list[str]:
        """Apply Stage-2 Unicode boost to *already-crafted* semantic passages.

        This is the preferred method for fair comparison experiments: it accepts
        the exact passages produced by :class:`PoisonedRAGAttack` and only adds
        the invisible-character DE perturbation on top.  The visible text is
        byte-for-byte identical to the input ``semantic_passages``.

        Use this instead of :meth:`craft_malicious_passages` when you want
        ``hybrid = semantic + Unicode boost`` (same passages, more retrieval).

        Args:
            target_question: Query to optimise toward.
            semantic_passages: Pre-crafted PoisonedRAG passages whose text
                content must not change.

        Returns:
            List of Unicode-perturbed passages, one per input passage, with
            ``strip_invisible(hybrid[i]) == semantic_passages[i]``.
        """
        if self._unicode.retriever is None:
            raise RuntimeError(
                "Set HybridAttack._unicode.retriever before calling boost_passages()."
            )
        query_emb: np.ndarray = self._unicode.retriever.encode_query(target_question)

        first = semantic_passages[0]
        start, end = self._locate_trigger_region(first)
        valid_positions = self._trigger_positions(first, start, end)

        _, shared_insertions = self._unicode._optimize_with_insertions(
            first, query_emb, valid_positions
        )
        logger.debug(
            "Hybrid boost_passages  DE done  %d insertions → applying to %d passages",
            len(shared_insertions),
            len(semantic_passages),
        )

        return [
            self._unicode._apply_perturbation(p, shared_insertions)
            for p in semantic_passages
        ]

    def inject(
        self,
        corpus: list[str],
        adversarial_passages: list[str],
    ) -> list[str]:
        """Inject hybrid passages into the corpus.

        Args:
            corpus: Original passage list (not mutated).
            adversarial_passages: Strings from :meth:`craft_malicious_passages`.

        Returns:
            New list: original passages first, adversarial passages appended.
        """
        return list(corpus) + list(adversarial_passages)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _locate_trigger_region(self, text: str) -> tuple[int, int]:
        """Return ``(start, end)`` char indices of the Unicode trigger region.

        ``"prefix"``      — first 50 characters (includes question prefix).
        ``"suffix"``      — last 50 characters (end of semantic content).
        ``"interleaved"`` — full passage range (whitespace positions used).

        Args:
            text: Stage-1 passage text.

        Returns:
            ``(start, end)`` index pair into *text* (end is inclusive-ish;
            passed directly to :meth:`_trigger_positions`).
        """
        if self.trigger_location == "suffix":
            return (max(0, len(text) - 50), len(text))
        if self.trigger_location == "interleaved":
            return (0, len(text))
        # Default: "prefix"
        return (0, min(50, len(text)))

    def _trigger_positions(self, text: str, start: int, end: int) -> list[int]:
        """Build the list of valid insertion positions within ``[start, end]``.

        For ``"interleaved"`` strategy, restricts to whitespace positions to
        avoid breaking word tokens.  Falls back to the full region if no
        whitespace found.

        Args:
            text: Stage-1 passage text.
            start: Region start (character index).
            end: Region end (character index, exclusive bound).

        Returns:
            Sorted list of valid insertion character indices.
        """
        end = min(end, len(text))
        full = list(range(start, end + 1))
        if self.trigger_location == "interleaved":
            ws = [i for i in range(start, end) if text[i] == " "]
            return ws if ws else full
        return full
