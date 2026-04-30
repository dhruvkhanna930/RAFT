"""Contriever dense retriever (facebook/contriever).

Default retriever matching the PoisonedRAG experimental setup.
Embeds queries and passages with mean-pooled Contriever hidden states,
then retrieves top-k by cosine similarity via a FAISS flat-IP index.

Models available:
- ``facebook/contriever``          — general Contriever
- ``facebook/contriever-msmarco``  — fine-tuned on MS-MARCO
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _autodetect_device() -> str:
    """Return the best available device: ``"cuda"``, ``"mps"``, or ``"cpu"``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class ContrieverRetriever:
    """Dense retriever backed by facebook/contriever.

    Encodes passages and queries with mean-pooled transformer hidden states.
    Maintains an in-memory FAISS flat-IP index after :meth:`build_index` is
    called.  For disk-cached indexing use
    :class:`~src.data.IndexBuilder` instead.

    Args:
        model_id: HuggingFace model ID (default ``"facebook/contriever"``).
        device: Inference device.  ``"auto"`` selects the best available.
        batch_size: Default encoding batch size.
        normalize: L2-normalise embeddings before indexing and querying.
    """

    def __init__(
        self,
        model_id: str = "facebook/contriever",
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = _autodetect_device() if device == "auto" else device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Any = None
        self._tokenizer: Any = None
        self._index: Any = None   # faiss.IndexFlatIP, set by build_index
        self._corpus: list[str] = []

    # ── Public encoding API ───────────────────────────────────────────────────

    def encode_passages(self, passages: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of passages into embedding vectors.

        Args:
            passages: Passage strings to encode.
            batch_size: Encoding batch size.

        Returns:
            Float32 array of shape ``(len(passages), dim)``.
        """
        return self._encode_batch(passages, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into an embedding vector.

        Args:
            query: Query string.

        Returns:
            Float32 array of shape ``(dim,)``.
        """
        return self._encode_batch([query], batch_size=1)[0]

    def load_index(self, index: Any, corpus: list[str]) -> None:
        """Inject a pre-built FAISS index, skipping the encode step.

        Use this when the index was previously built by
        :class:`~src.data.IndexBuilder` and saved to disk — avoids
        re-encoding the corpus on every script run.

        Args:
            index: A FAISS index whose vectors correspond to *corpus*.
            corpus: Passage strings parallel to the index vectors.
        """
        self._index = index
        self._corpus = list(corpus)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* — alias for ``encode_passages`` with default batch size.

        Args:
            texts: List of strings to encode.

        Returns:
            Float32 array of shape ``(len(texts), dim)``.
        """
        return self._encode_batch(texts, batch_size=self.batch_size)

    # ── RagBase-compatible interface ──────────────────────────────────────────

    def build_index(self, corpus: list[str]) -> None:
        """Encode *corpus* and build an in-memory FAISS flat-IP index.

        Called by :meth:`~src.rag.base.RagBase.load_corpus`.

        Args:
            corpus: List of passage strings to index.
        """
        import faiss

        logger.info("Building in-memory index over %d passages…", len(corpus))
        embeddings = self.encode_passages(corpus, batch_size=self.batch_size).astype(
            np.float32
        )
        if self.normalize:
            emb = embeddings.copy()
            faiss.normalize_L2(emb)
        else:
            emb = embeddings

        dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(emb)
        self._corpus = list(corpus)
        logger.info("Index ready: %d vectors, dim=%d", self._index.ntotal, dim)

    def retrieve(self, query: str, k: int = 5) -> tuple[list[str], list[float]]:
        """Return top-*k* passages for *query*.

        :meth:`build_index` must have been called first.

        Args:
            query: Query string.
            k: Number of passages to return.

        Returns:
            Tuple of ``(passages, scores)`` sorted by descending cosine score.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        import faiss

        if self._index is None or not self._corpus:
            raise RuntimeError("Call build_index(corpus) before retrieve().")

        query_vec = self.encode_query(query).astype(np.float32).reshape(1, -1).copy()
        if self.normalize:
            faiss.normalize_L2(query_vec)
        scores, indices = self._index.search(query_vec, k)
        top_passages = [
            self._corpus[i] for i in indices[0] if 0 <= i < len(self._corpus)
        ]
        top_scores = [float(s) for s in scores[0][: len(top_passages)]]
        return top_passages, top_scores

    # ── Private ───────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Mean-pool last hidden state in batches.

        Args:
            texts: Strings to encode.
            batch_size: Batch size.

        Returns:
            Float32 array of shape ``(len(texts), dim)``.
        """
        import torch
        import torch.nn.functional as F

        self._load_model()
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if self.normalize:
                emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def _load_model(self) -> None:
        """Lazy-load the Contriever tokenizer and model from HuggingFace."""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading %s onto %s…", self.model_id, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    @staticmethod
    def _mean_pool(
        token_embeddings: Any,
        attention_mask: Any,
    ) -> Any:
        """Mean-pool *token_embeddings* weighted by *attention_mask*.

        Args:
            token_embeddings: Shape ``(batch, seq_len, dim)``.
            attention_mask: Shape ``(batch, seq_len)``.

        Returns:
            Shape ``(batch, dim)`` mean-pooled tensor.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
