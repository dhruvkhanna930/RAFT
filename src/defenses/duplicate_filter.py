"""Duplicate / near-duplicate passage filter.

Tests whether deduplication neutralises attacks that inject multiple copies
of the same adversarial passage (up to ``injection_budget`` copies).
"""

from __future__ import annotations

from src.defenses.base import DefenseBase


class DuplicateFilter(DefenseBase):
    """Remove exact and near-duplicate passages, preserving first-occurrence order.

    Satisfies :class:`DefenseBase`: ``apply`` accepts a single string or a
    list of strings.  A single string has no duplicates to remove and is
    returned unchanged.

    Args:
        exact_only: If True, only remove exact string duplicates (O(n)).
            If False, also remove near-duplicates by Jaccard similarity (O(n²)).
        jaccard_threshold: Minimum Jaccard similarity to treat as near-duplicate.
        ngram_size: N-gram size for Jaccard computation.
    """

    def __init__(
        self,
        exact_only: bool = False,
        jaccard_threshold: float = 0.9,
        ngram_size: int = 3,
    ) -> None:
        self.exact_only = exact_only
        self.jaccard_threshold = jaccard_threshold
        self.ngram_size = ngram_size

    # ── DefenseBase interface ─────────────────────────────────────────────────

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Remove duplicate passages.

        Args:
            text_or_passages: Single passage string (returned unchanged), or a
                list of passages to deduplicate.

        Returns:
            Cleaned string or deduplicated list.
        """
        if isinstance(text_or_passages, str):
            return text_or_passages
        return self._deduplicate(text_or_passages)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _deduplicate(self, passages: list[str]) -> list[str]:
        seen: list[str] = []
        seen_set: set[str] = set()
        for p in passages:
            if p in seen_set:
                continue
            if not self.exact_only:
                if any(self._jaccard(p, s) >= self.jaccard_threshold for s in seen):
                    continue
            seen.append(p)
            seen_set.add(p)
        return seen

    def _ngrams(self, text: str) -> set[tuple[str, ...]]:
        """Compute the token n-gram set of *text*.

        Args:
            text: Input string (whitespace-tokenised).

        Returns:
            Set of n-gram tuples.
        """
        tokens = text.lower().split()
        n = self.ngram_size
        if len(tokens) < n:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def _jaccard(self, a: str, b: str) -> float:
        """Jaccard similarity between the n-gram sets of *a* and *b*.

        Args:
            a: First passage string.
            b: Second passage string.

        Returns:
            Jaccard similarity in [0, 1].
        """
        set_a = self._ngrams(a)
        set_b = self._ngrams(b)
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
