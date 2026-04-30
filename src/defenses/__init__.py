"""Defense modules — post-retrieval and corpus-level countermeasures."""

from src.defenses.base import DefenseBase, DefenseChain
from src.defenses.duplicate_filter import DuplicateFilter
from src.defenses.paraphrase import ParaphraseDefense, QueryParaphraseDefense
from src.defenses.perplexity import PerplexityFilter
from src.defenses.unicode_normalize import UnicodeNormalizer
from src.defenses.zero_width_strip import ZeroWidthStripDefense

__all__ = [
    "DefenseBase",
    "DefenseChain",
    "DuplicateFilter",
    "ParaphraseDefense",
    "QueryParaphraseDefense",
    "PerplexityFilter",
    "UnicodeNormalizer",
    "ZeroWidthStripDefense",
]
