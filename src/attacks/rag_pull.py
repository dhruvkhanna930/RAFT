"""RAG-Pull Unicode perturbation attack.

Replicates the invisible-character retrieval-boosting attack from RAG-Pull:
insert zero-width and other invisible Unicode characters into a passage to
shift its embedding closer to a target query — no visible text change.
Optimisation uses Differential Evolution (DE) from SciPy.

Reference: RAG-Pull (arXiv, 2024)

Three operation modes (``insertion_strategy``):

- ``"whitespace"``  — insertions are anchored to whitespace positions only.
- ``"boundary"``    — insertions at word-boundary positions.
- ``"random"``      — insertions at any character position.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from src.attacks.base import AttackBase, AttackConfig
from src.attacks.unicode_chars import (
    CharCategory,
    UnicodeInventory,
    perturb,
)

logger = logging.getLogger(__name__)

# Default char inventory for RAGPullAttack — includes variation selectors
# because they are Unicode category Mn (non-spacing mark) and are more likely
# to survive BERT-style tokeniser normalisation than Cf format characters.
_DEFAULT_CATEGORIES = [
    CharCategory.ZERO_WIDTH_JOINER,
    CharCategory.ZERO_WIDTH_NON_JOINER,
    CharCategory.ZERO_WIDTH_SPACE,
    CharCategory.WORD_JOINER,
    CharCategory.SOFT_HYPHEN,
    CharCategory.TAG_CHARS,
    CharCategory.VARIATION_SELECTORS,
]


class RAGPullAttack(AttackBase):
    """DE-based Unicode perturbation attack.

    Decision variables: positions + character identities of invisible
    insertions.  Fitness function: cosine similarity between the perturbed
    passage embedding and the query embedding.

    Args:
        config: Shared ``AttackConfig``.
        retriever: Retriever instance with an ``embed(texts) -> np.ndarray``
            method.  Must be set before calling
            :meth:`craft_malicious_passages`.
        char_inventory: Curated invisible character set.  Defaults to all
            categories including variation selectors.
        perturbation_budget: Maximum invisible chars to insert per passage.
        de_population: DE population-size multiplier (``popsize`` in SciPy).
            Actual population = ``de_population × 2 × perturbation_budget``.
        de_max_iter: Maximum DE generations.
        de_mutation: DE mutation factor *F*.
        de_crossover: DE crossover probability *CR*.
        insertion_strategy: ``"whitespace"``, ``"boundary"``, or ``"random"``.
    """

    def __init__(
        self,
        config: AttackConfig,
        retriever: Any | None = None,
        char_inventory: UnicodeInventory | None = None,
        perturbation_budget: int = 50,
        de_population: int = 20,
        de_max_iter: int = 100,
        de_mutation: float = 0.8,
        de_crossover: float = 0.9,
        insertion_strategy: str = "whitespace",
    ) -> None:
        super().__init__(config)
        self.retriever = retriever
        self.char_inventory = char_inventory or UnicodeInventory(
            categories=_DEFAULT_CATEGORIES
        )
        self.perturbation_budget = perturbation_budget
        self.de_population = de_population
        self.de_max_iter = de_max_iter
        self.de_mutation = de_mutation
        self.de_crossover = de_crossover
        self.insertion_strategy = insertion_strategy

    # ── Public API ─────────────────────────────────────────────────────────────

    def craft_malicious_passages(
        self,
        target_question: str,
        target_answer: str,
        n: int,
    ) -> list[str]:
        """Apply Unicode perturbation to boost retrieval rank for *n* passages.

        Builds a base passage containing the target answer, then runs
        Differential Evolution to find invisible-character insertions that
        maximise cosine similarity between the perturbed passage and the
        target query embedding.

        Args:
            target_question: Query whose embedding we optimise toward.
            target_answer: Used to construct the base passage text.
            n: Number of perturbed passages to produce.

        Returns:
            List of *n* Unicode-perturbed passage strings (all identical —
            the DE finds one optimal perturbation).

        Raises:
            RuntimeError: If ``self.retriever`` has not been set.
        """
        if self.retriever is None:
            raise RuntimeError(
                "RAGPullAttack.retriever must be set before calling "
                "craft_malicious_passages()."
            )
        if len(self.char_inventory) == 0:
            raise ValueError("char_inventory is empty — nothing to insert.")

        base_text = self._build_base_passage(target_answer)
        query_emb = self.retriever.encode_query(target_question)

        # Log initial cosine similarity before optimisation.
        base_emb = self.retriever.embed([base_text])[0]
        sim_before = float(np.dot(base_emb, query_emb))
        logger.info(
            "RAG-Pull  question=%r  base_sim=%.4f  budget=%d  maxiter=%d",
            target_question[:60],
            sim_before,
            self.perturbation_budget,
            self.de_max_iter,
        )

        best_text = self._optimize(base_text, query_emb)

        best_emb = self.retriever.embed([best_text])[0]
        sim_after = float(np.dot(best_emb, query_emb))
        logger.info(
            "RAG-Pull  sim_after=%.4f  Δsim=+%.4f",
            sim_after,
            sim_after - sim_before,
        )

        return [best_text] * n

    def inject(
        self,
        corpus: list[str],
        adversarial_passages: list[str],
    ) -> list[str]:
        """Append Unicode-perturbed passages to the corpus.

        Args:
            corpus: Original passage list.
            adversarial_passages: Strings from :meth:`craft_malicious_passages`.

        Returns:
            Poisoned corpus (original passages first, adversarial appended).
        """
        return list(corpus) + list(adversarial_passages)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_base_passage(target_answer: str) -> str:
        """Construct the plain-text base passage from *target_answer*.

        Args:
            target_answer: The desired (fake) answer to embed.

        Returns:
            Multi-sentence passage string.
        """
        return (
            f"According to recent information, {target_answer} is the correct answer. "
            f"{target_answer} has been confirmed by multiple reliable sources. "
            f"The most accurate and verified answer is {target_answer}."
        )

    def _valid_positions(self, text: str) -> list[int]:
        """Return valid insertion positions based on ``insertion_strategy``.

        Args:
            text: Base passage text.

        Returns:
            Sorted list of character indices where invisible chars may be
            inserted.  Falls back to ``[0 … len(text)]`` if no positions
            satisfy the strategy constraint.
        """
        if self.insertion_strategy == "whitespace":
            positions = [i for i, c in enumerate(text) if c == " "]
            if positions:
                return positions
        elif self.insertion_strategy == "boundary":
            positions = sorted(
                {m.start() for m in re.finditer(r"\b", text)}
            )
            if positions:
                return positions
        # "random" or fallback
        return list(range(len(text) + 1))

    def _decode_individual(
        self, x: np.ndarray, valid_positions: list[int]
    ) -> list[tuple[int, str]]:
        """Map a DE individual (flat float vector) to a list of insertions.

        Args:
            x: 1-D array of shape ``(2 × perturbation_budget,)``.  Even
                indices are position fractions; odd indices are character
                fractions, both in ``[0, 1]``.
            valid_positions: Allowed insertion positions (from
                :meth:`_valid_positions`).

        Returns:
            List of ``(position, char)`` tuples suitable for
            :func:`~src.attacks.unicode_chars.perturb`.
        """
        n_chars = len(self.char_inventory)
        n_pos = len(valid_positions)
        insertions: list[tuple[int, str]] = []
        for i in range(self.perturbation_budget):
            pos_frac = float(x[2 * i])
            char_frac = float(x[2 * i + 1])
            pos_idx = int(pos_frac * n_pos)
            pos_idx = max(0, min(pos_idx, n_pos - 1))
            pos = valid_positions[pos_idx]
            char_idx = int(char_frac * n_chars) % n_chars
            char = self.char_inventory.chars[char_idx]
            insertions.append((pos, char))
        return insertions

    def perturb_passage(
        self,
        passage: str,
        query_emb: np.ndarray,
        valid_positions: list[int] | None = None,
    ) -> str:
        """Apply DE perturbation to an externally-provided passage.

        Used by :class:`~src.attacks.hybrid.HybridAttack` to apply Unicode
        perturbation on top of a PoisonedRAG semantic payload.

        Args:
            passage: Base passage text (e.g. a PoisonedRAG semantic payload).
            query_emb: Pre-computed L2-normalised query embedding.
            valid_positions: Optional override for valid insertion positions.
                If ``None``, determined by ``self.insertion_strategy``.

        Returns:
            Perturbed passage string.

        Raises:
            RuntimeError: If ``self.retriever`` is ``None``.
        """
        if self.retriever is None:
            raise RuntimeError(
                "RAGPullAttack.retriever must be set before calling perturb_passage()."
            )
        return self._optimize(passage, query_emb, valid_positions=valid_positions)

    def _optimize(
        self,
        base_text: str,
        query_emb: np.ndarray,
        valid_positions: list[int] | None = None,
    ) -> str:
        """Run Differential Evolution to find the best insertion set.

        Minimises ``-cosine_similarity(embed(perturbed), query_emb)`` over
        ``2 * perturbation_budget`` continuous decision variables.

        Args:
            base_text: Unperturbed passage.
            query_emb: Pre-computed, L2-normalised query embedding.
            valid_positions: Optional list of character indices to restrict
                insertions to (e.g. only within a trigger region).  If
                ``None``, :meth:`_valid_positions` is called.

        Returns:
            Optimally perturbed passage string.
        """
        text, _ = self._optimize_with_insertions(base_text, query_emb, valid_positions)
        return text

    def _optimize_with_insertions(
        self,
        base_text: str,
        query_emb: np.ndarray,
        valid_positions: list[int] | None = None,
    ) -> tuple[str, list[tuple[int, str]]]:
        """Run DE optimisation, returning both the perturbed text and the raw insertions.

        Unlike :meth:`_optimize`, this also returns the ``(position, char)``
        insertion list so callers can re-apply the same perturbation to other
        passage texts without running DE again.  This is used by
        :class:`~src.attacks.hybrid.HybridAttack` to share one DE run across
        all *n* semantic passages.

        Args:
            base_text: Unperturbed passage.
            query_emb: Pre-computed, L2-normalised query embedding.
            valid_positions: Optional override for valid insertion positions.

        Returns:
            Tuple ``(best_text, best_insertions)`` where ``best_insertions``
            is a list of ``(char_index, char)`` tuples.
        """
        if valid_positions is None:
            valid_positions = self._valid_positions(base_text)
        bounds = [(0.0, 1.0), (0.0, 1.0)] * self.perturbation_budget

        retriever = self.retriever  # local ref for closure

        # Vectorized objective: scipy passes the whole population as a 2-D array
        # x of shape (n_params, pop_size) so we embed all candidates in one
        # batched retriever.embed() call instead of n_params × pop_size serial
        # calls.  This is the dominant speedup — ~50-100× on MPS/CUDA because
        # BERT forward passes are heavily parallelised in a batch.
        def _objective_vec(x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                x = x[:, None]
            pop_size = x.shape[1]
            texts = [
                self._apply_perturbation(
                    base_text,
                    self._decode_individual(x[:, i], valid_positions),
                )
                for i in range(pop_size)
            ]
            embs = np.array(retriever.embed(texts))  # (pop_size, dim)
            return -(embs @ query_emb)               # (pop_size,) — minimise negative sim

        result = differential_evolution(
            _objective_vec,
            bounds,
            popsize=self.de_population,
            maxiter=self.de_max_iter,
            mutation=self.de_mutation,
            recombination=self.de_crossover,
            seed=42,
            tol=1e-7,
            polish=False,
            disp=False,
            vectorized=True,   # batch-embed entire population per generation
        )

        best_insertions = self._decode_individual(result.x, valid_positions)
        best_text = self._apply_perturbation(base_text, best_insertions)
        return best_text, best_insertions

    def _apply_perturbation(
        self, text: str, insertions: list[tuple[int, str]]
    ) -> str:
        """Insert invisible characters at the given positions.

        Delegates to :func:`~src.attacks.unicode_chars.perturb` which
        applies insertions right-to-left to avoid index shifting.

        Args:
            text: Base passage text.
            insertions: ``(position, char)`` pairs.

        Returns:
            Perturbed text string.
        """
        return perturb(text, insertions)

    def _fitness(
        self,
        perturbed_text: str,
        query_embedding: np.ndarray,
        retriever: Any,
    ) -> float:
        """Cosine similarity between perturbed passage and query embeddings.

        Used for standalone evaluation; DE optimisation calls
        :meth:`_optimize` directly.

        Args:
            perturbed_text: Passage after Unicode insertion.
            query_embedding: Pre-computed L2-normalised query embedding.
            retriever: Retriever with an ``embed()`` method.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        emb = retriever.embed([perturbed_text])[0]
        return float(np.dot(emb, query_embedding))
