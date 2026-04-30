"""FAISS index builder with disk caching.

Encodes a passage corpus with any retriever that exposes
``encode_passages(passages, batch_size) -> np.ndarray``, builds a FAISS
flat-IP index, and caches both the index and the corpus to disk so that
re-embedding is skipped on subsequent runs.

Cache layout::

    data/indices/
      nq_contriever_1000.faiss         ← FAISS binary
      nq_contriever_1000.corpus.jsonl  ← passages (one JSON per line)
      nq_contriever_full.faiss
      nq_contriever_full.corpus.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class _SupportsEncode(Protocol):
    """Structural type for any retriever used by IndexBuilder."""

    def encode_passages(self, passages: list[str], batch_size: int = 64) -> np.ndarray:
        ...

    def encode_query(self, query: str) -> np.ndarray:
        ...


class IndexBuilder:
    """Build and cache a FAISS flat-IP index from a passage corpus.

    Args:
        indices_dir: Directory to store cached index files.
        dataset: Dataset name (used in cache filename, e.g. ``"nq"``).
        retriever_name: Retriever identifier (e.g. ``"contriever"``).
    """

    def __init__(
        self,
        indices_dir: str | Path = "data/indices",
        dataset: str = "nq",
        retriever_name: str = "contriever",
    ) -> None:
        self.indices_dir = Path(indices_dir)
        self.dataset = dataset
        self.retriever_name = retriever_name

    # ── Cache filenames ───────────────────────────────────────────────────────

    def cache_key(self, corpus_size: int) -> str:
        """Return the cache key for this corpus size.

        Args:
            corpus_size: Number of passages; use ``-1`` for the full corpus.

        Returns:
            String like ``"nq_contriever_1000"`` or ``"nq_contriever_full"``.
        """
        size_str = "full" if corpus_size == -1 else str(corpus_size)
        return f"{self.dataset}_{self.retriever_name}_{size_str}"

    def index_path(self, corpus_size: int) -> Path:
        """Path to the cached FAISS binary for *corpus_size*."""
        return self.indices_dir / f"{self.cache_key(corpus_size)}.faiss"

    def corpus_path(self, corpus_size: int) -> Path:
        """Path to the cached corpus JSONL for *corpus_size*."""
        return self.indices_dir / f"{self.cache_key(corpus_size)}.corpus.jsonl"

    # ── Build / load ──────────────────────────────────────────────────────────

    def build(
        self,
        passages: list[str],
        retriever: _SupportsEncode,
        batch_size: int = 64,
        force: bool = False,
    ) -> tuple[faiss.Index, list[str]]:
        """Encode *passages* and build a FAISS flat-IP index.

        Loads from disk cache if available and ``force`` is False.

        Args:
            passages: Passage strings to encode and index.
            retriever: Encoder with ``encode_passages`` and ``encode_query``.
            batch_size: Batch size forwarded to ``encode_passages``.
            force: Rebuild even if a cached index exists.

        Returns:
            Tuple of ``(faiss_index, passages)``.
        """
        n = len(passages)
        idx_path = self.index_path(n)
        corp_path = self.corpus_path(n)

        if not force and idx_path.exists() and corp_path.exists():
            logger.info("Loading cached index from %s", idx_path)
            index = faiss.read_index(str(idx_path))
            loaded = _load_corpus_jsonl(corp_path)
            return index, loaded

        self.indices_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Encoding %d passages with %s …", n, self.retriever_name
        )
        embeddings: np.ndarray = retriever.encode_passages(passages, batch_size=batch_size)
        embeddings = embeddings.astype(np.float32)
        emb_normed = embeddings.copy()
        faiss.normalize_L2(emb_normed)

        dim = emb_normed.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_normed)

        faiss.write_index(index, str(idx_path))
        _save_corpus_jsonl(passages, corp_path)
        logger.info(
            "Index saved → %s  (%d vectors, dim=%d)", idx_path, index.ntotal, dim
        )
        return index, passages

    def load(self, corpus_size: int) -> tuple[faiss.Index, list[str]]:
        """Load a previously built index from disk cache.

        Args:
            corpus_size: Corpus size used when the index was built
                (``-1`` for the full corpus).

        Returns:
            Tuple of ``(faiss_index, passages)``.

        Raises:
            FileNotFoundError: If no cached index exists for *corpus_size*.
        """
        idx_path = self.index_path(corpus_size)
        corp_path = self.corpus_path(corpus_size)
        if not idx_path.exists():
            raise FileNotFoundError(
                f"No cached index at {idx_path}. "
                "Run IndexBuilder.build() first."
            )
        index = faiss.read_index(str(idx_path))
        passages = _load_corpus_jsonl(corp_path)
        return index, passages

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        index: faiss.Index,
        passages: list[str],
        k: int = 5,
    ) -> tuple[list[str], list[float]]:
        """Search *index* for the nearest passages to *query_vec*.

        Args:
            query_vec: 1-D float32 query embedding (will be L2-normalised).
            index: FAISS index to search (must be a flat-IP index).
            passages: Passage strings parallel to the index vectors.
            k: Number of results to return.

        Returns:
            Tuple of ``(top-k passages, scores)`` sorted by descending score.
        """
        vec = query_vec.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        scores, indices = index.search(vec, k)
        top_passages = [
            passages[i] for i in indices[0] if 0 <= i < len(passages)
        ]
        top_scores = [float(s) for s in scores[0][: len(top_passages)]]
        return top_passages, top_scores


# ── JSONL helpers (module-level, not part of the class) ──────────────────────


def _save_corpus_jsonl(passages: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")


def _load_corpus_jsonl(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh]
