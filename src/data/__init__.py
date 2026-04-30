"""Data loading utilities.

Re-exports:
    NQLoader     — download and cache the NQ corpus + questions from HuggingFace
    IndexBuilder — build and persist a FAISS index for a passage corpus
"""

from src.data.index_builder import IndexBuilder
from src.data.nq_loader import NQLoader

__all__ = ["NQLoader", "IndexBuilder"]
