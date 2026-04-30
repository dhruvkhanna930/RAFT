"""Curated inventory of invisible / imperceptible Unicode characters.

Provides the character sets used by RAGPullAttack and HybridAttack.
Each category contains characters that:
- are invisible when rendered in a standard monospace font,
- have Unicode codepoints that most embedding tokenisers treat as
  distinct tokens (shifting the resulting embedding vector),
- survive copy-paste without modification.

Character categories (from Bad Characters, Boucher et al. S&P 2022,
and the RAG-Pull supplementary material):
- Zero-width joiners / non-joiners
- Zero-width spaces / word joiners
- Soft hyphens
- Unicode tag characters (U+E0000 block)
- Variation selectors
- Bidirectional control characters (with caution — may be visually detectable)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class CharCategory(str, Enum):
    """Enumeration of supported invisible character categories."""

    ZERO_WIDTH_JOINER = "zero_width_joiner"
    ZERO_WIDTH_NON_JOINER = "zero_width_non_joiner"
    ZERO_WIDTH_SPACE = "zero_width_space"
    WORD_JOINER = "word_joiner"
    SOFT_HYPHEN = "soft_hyphen"
    TAG_CHARS = "tag_chars"
    VARIATION_SELECTORS = "variation_selectors"


# Character maps — codepoint → description
ZERO_WIDTH_CHARS: dict[str, str] = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u2060": "WORD JOINER",
    "\u2061": "FUNCTION APPLICATION",
    "\u2062": "INVISIBLE TIMES",
    "\u2063": "INVISIBLE SEPARATOR",
    "\u2064": "INVISIBLE PLUS",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE (BOM)",
    "\u00ad": "SOFT HYPHEN",
    "\u034f": "COMBINING GRAPHEME JOINER",
    "\u180e": "MONGOLIAN VOWEL SEPARATOR",
}

# Unicode tag block U+E0001–U+E007F
# Each char encodes one ASCII character: chr(0xE0000 + ord(ascii_char))
# e.g. chr(0xE0061) = tag 'a', chr(0xE0020) = tag SPACE
TAG_CHARS: dict[str, str] = {
    chr(cp): f"TAG {chr(cp - 0xE0000)!r}"
    for cp in range(0xE0001, 0xE0080)
}

# Variation selectors U+FE00–U+FE0F (VS1–VS16)
VARIATION_SELECTORS: dict[str, str] = {
    chr(cp): f"VARIATION SELECTOR-{cp - 0xFE00 + 1}"
    for cp in range(0xFE00, 0xFE10)
}

# Category → char-dict mapping used by UnicodeInventory
_CATEGORY_MAP: dict[CharCategory, dict[str, str]] = {
    CharCategory.ZERO_WIDTH_SPACE: {"\u200b": ZERO_WIDTH_CHARS["\u200b"]},
    CharCategory.ZERO_WIDTH_NON_JOINER: {"\u200c": ZERO_WIDTH_CHARS["\u200c"]},
    CharCategory.ZERO_WIDTH_JOINER: {"\u200d": ZERO_WIDTH_CHARS["\u200d"]},
    CharCategory.WORD_JOINER: {
        k: v
        for k, v in ZERO_WIDTH_CHARS.items()
        if k in {"\u2060", "\u2061", "\u2062", "\u2063", "\u2064", "\ufeff", "\u034f", "\u180e"}
    },
    CharCategory.SOFT_HYPHEN: {"\u00ad": ZERO_WIDTH_CHARS["\u00ad"]},
    CharCategory.TAG_CHARS: TAG_CHARS,
    CharCategory.VARIATION_SELECTORS: VARIATION_SELECTORS,
}

# Flat set of ALL known invisible chars (for strip_invisible)
_ALL_INVISIBLE: frozenset[str] = frozenset(
    list(ZERO_WIDTH_CHARS) + list(TAG_CHARS) + list(VARIATION_SELECTORS)
)


@dataclass
class UnicodeInventory:
    """Collection of invisible characters available for perturbation.

    Args:
        categories: Which ``CharCategory`` values to include.  Defaults to
            all except bidi controls (which may be visually detectable).
    """

    categories: list[CharCategory] = field(
        default_factory=lambda: [
            CharCategory.ZERO_WIDTH_JOINER,
            CharCategory.ZERO_WIDTH_NON_JOINER,
            CharCategory.ZERO_WIDTH_SPACE,
            CharCategory.WORD_JOINER,
            CharCategory.SOFT_HYPHEN,
            CharCategory.TAG_CHARS,
        ]
    )
    _chars: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the flat character list from selected categories."""
        seen: set[str] = set()
        for cat in self.categories:
            for ch in _CATEGORY_MAP.get(cat, {}).keys():
                if ch not in seen:
                    seen.add(ch)
                    self._chars.append(ch)

    @property
    def chars(self) -> list[str]:
        """Flat list of available invisible characters.

        Returns:
            List of single-character strings.
        """
        return self._chars

    def __len__(self) -> int:
        return len(self._chars)

    def is_invisible(self, char: str) -> bool:
        """Return True if *char* is in this inventory.

        Args:
            char: Single Unicode character.

        Returns:
            True if the character is in the inventory.
        """
        return char in set(self._chars)

    @staticmethod
    def strip_invisible(text: str) -> str:
        """Remove all known invisible characters from *text* (NFKC-style defence).

        Strips every character in ``_ALL_INVISIBLE`` then applies NFKC
        normalisation.

        Args:
            text: Input string (potentially perturbed).

        Returns:
            Cleaned string with all invisible chars removed.
        """
        for ch in _ALL_INVISIBLE:
            if ch in text:
                text = text.replace(ch, "")
        return unicodedata.normalize("NFKC", text)


def perturb(text: str, insertions: list[tuple[int, str]]) -> str:
    """Insert invisible characters at specified positions into *text*.

    Applies insertions right-to-left to avoid index shifting — each
    ``position`` refers to an index in the *original* ``text``, not the
    partially-modified string.

    Args:
        text: Base text string.
        insertions: List of ``(position, char)`` pairs.  ``position`` is a
            0-based character index; values outside ``[0, len(text)]`` are
            clamped to the valid range.

    Returns:
        New string with all invisible characters inserted.
    """
    result = list(text)
    for pos, char in sorted(insertions, key=lambda t: -t[0]):
        pos = max(0, min(pos, len(result)))
        result.insert(pos, char)
    return "".join(result)
