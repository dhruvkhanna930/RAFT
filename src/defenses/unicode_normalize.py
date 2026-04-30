"""Unicode normalisation defense — NFKC + zero-width character stripping.

The strongest defense specifically targeting RAG-Pull-style attacks.
Applying this defense at corpus index time should neutralise the embedding
perturbation entirely (hypothesis to verify in Phase 4).
"""

from __future__ import annotations

import unicodedata

from src.defenses.base import DefenseBase

# Zero-width / format characters that NFKC does not always eliminate
_ZW_STRIP = frozenset([
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\u2061",  # FUNCTION APPLICATION
    "\u2062",  # INVISIBLE TIMES
    "\u2063",  # INVISIBLE SEPARATOR
    "\u2064",  # INVISIBLE PLUS
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u00ad",  # SOFT HYPHEN
    "\u034f",  # COMBINING GRAPHEME JOINER
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR
])

# Unicode tag block U+E0001–U+E007F
_TAG_STRIP = frozenset(chr(cp) for cp in range(0xE0001, 0xE0080))

# Variation selectors U+FE00–U+FE0F
_VS_STRIP = frozenset(chr(cp) for cp in range(0xFE00, 0xFE10))

_ALL_STRIP = _ZW_STRIP | _TAG_STRIP | _VS_STRIP


class UnicodeNormalizer(DefenseBase):
    """Strip invisible characters and apply NFKC normalisation.

    Satisfies :class:`DefenseBase`: ``apply`` accepts a single string or a
    list of strings and returns the same type.

    Args:
        nfkc: Whether to apply NFKC normalisation.
        strip_zero_width: Strip zero-width / format chars.
        strip_tags: Strip Unicode tag block (U+E0001–U+E007F).
        strip_variation_selectors: Strip variation selectors (U+FE00–U+FE0F).
    """

    def __init__(
        self,
        nfkc: bool = True,
        strip_zero_width: bool = True,
        strip_tags: bool = True,
        strip_variation_selectors: bool = True,
    ) -> None:
        self.nfkc = nfkc
        self.strip_zero_width = strip_zero_width
        self.strip_tags = strip_tags
        self.strip_variation_selectors = strip_variation_selectors

        active: frozenset[str] = frozenset()
        if strip_zero_width:
            active |= _ZW_STRIP
        if strip_tags:
            active |= _TAG_STRIP
        if strip_variation_selectors:
            active |= _VS_STRIP
        self._strip = active

    # ── DefenseBase interface ─────────────────────────────────────────────────

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Remove invisible Unicode characters from a passage or list.

        Args:
            text_or_passages: Single passage string, or a list of passages.

        Returns:
            Cleaned string if input was a string; cleaned list otherwise.
        """
        if isinstance(text_or_passages, str):
            return self.clean(text_or_passages)
        return [self.clean(p) for p in text_or_passages]

    # ── Convenience ───────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Remove invisible characters from a single string.

        Args:
            text: Input string.

        Returns:
            Cleaned string.
        """
        if self._strip:
            text = "".join(ch for ch in text if ch not in self._strip)
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        return text
