"""Zero-width and bidi-control character strip defense.

Pure-regex removal — faster and simpler than full NFKC normalisation.
Targets exactly the character classes used by RAG-Pull-style embedding-
boosting attacks: zero-width characters, Unicode tag block (U+E0001–U+E007F),
variation selectors (U+FE00–U+FE0F), and bidirectional control characters.

Unlike :class:`~src.defenses.unicode_normalize.UnicodeNormalizer`, this
defense does **not** apply NFKC decomposition, so visible characters such as
ligatures and full-width forms are left intact.  Use this when you want the
minimal, fastest possible strip with no risk of changing display-visible text.
"""

from __future__ import annotations

import re

from src.defenses.base import DefenseBase

# Pre-compiled pattern covering all character classes targeted by RAG-Pull.
# Character ranges listed explicitly so the intent is auditable.
_STRIP_RE = re.compile(
    "["
    "\u00ad"                    # SOFT HYPHEN
    "\u034f"                    # COMBINING GRAPHEME JOINER
    "\u180e"                    # MONGOLIAN VOWEL SEPARATOR
    "\u200b-\u200f"             # ZW SPACE / ZW NON-JOINER / ZW JOINER / LRM / RLM
    "\u202a-\u202e"             # LRE / RLE / PDF / LRO / RLO  (bidi controls)
    "\u2060-\u2064"             # WORD JOINER + invisible math operators
    "\u206a-\u206f"             # deprecated format characters
    "\ufeff"                    # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\ufe00-\ufe0f"             # variation selectors VS-1 – VS-16
    "\U000E0001-\U000E007F"     # Unicode tag block
    "]"
)


class ZeroWidthStripDefense(DefenseBase):
    """Strip zero-width and bidi-control characters using a precompiled regex.

    Satisfies :class:`DefenseBase`: ``apply`` accepts a single string or a
    list of strings and returns the same type.

    Args:
        strip_variation_selectors: Remove variation selectors U+FE00–U+FE0F.
            Default ``True``.
        strip_tag_block: Remove Unicode tag-block chars U+E0001–U+E007F.
            Default ``True``.
    """

    def __init__(
        self,
        strip_variation_selectors: bool = True,
        strip_tag_block: bool = True,
    ) -> None:
        self.strip_variation_selectors = strip_variation_selectors
        self.strip_tag_block = strip_tag_block
        # Use the module-level compiled pattern for the default (both True).
        if strip_variation_selectors and strip_tag_block:
            self._re = _STRIP_RE
        else:
            base = (
                "\u00ad\u034f\u180e"
                "\u200b-\u200f"
                "\u202a-\u202e"
                "\u2060-\u2064"
                "\u206a-\u206f"
                "\ufeff"
            )
            extras = ""
            if strip_variation_selectors:
                extras += "\ufe00-\ufe0f"
            if strip_tag_block:
                extras += "\U000E0001-\U000E007F"
            self._re = re.compile("[" + base + extras + "]")

    # ── DefenseBase interface ─────────────────────────────────────────────────

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Strip invisible characters from a passage or list of passages.

        Args:
            text_or_passages: Single passage string or list of passages.

        Returns:
            Cleaned string if input was a string; cleaned list otherwise.
        """
        if isinstance(text_or_passages, str):
            return self.strip(text_or_passages)
        return [self.strip(p) for p in text_or_passages]

    # ── Convenience ───────────────────────────────────────────────────────────

    def strip(self, text: str) -> str:
        """Remove all targeted invisible characters from *text*.

        Args:
            text: Input string.

        Returns:
            String with zero-width / bidi-control characters removed.
        """
        return self._re.sub("", text)
