"""BGE dense retriever (BAAI/bge-base-en-v1.5).

Alternative retriever for ablation studies.  BGE uses a different
embedding space than Contriever, allowing us to test whether Unicode
attacks transfer across embedders.

Query prefix: BGE requires prepending "Represent this sentence for searching
relevant passages: " to queries (not to passages).
"""

from __future__ import annotations

import numpy as np


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGERetriever:
    """Dense retriever backed by BAAI/bge-base-en-v1.5.

    Identical interface to ``ContrieverRetriever``.  Differs in the model
    used and the required query prefix.

    Args:
        model_id: HuggingFace model ID.
        device: ``"cpu"`` or ``"cuda"``.
        batch_size: Encoding batch size.
        normalize: Whether to L2-normalise embeddings.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._tokenizer = None
        self._index = None
        self._corpus: list[str] = []

    def build_index(self, corpus: list[str]) -> None:
        """Encode *corpus* and build a FAISS flat-IP index.

        Args:
            corpus: List of passage strings to index.
        """
        # TODO: same as ContrieverRetriever.build_index but with BGE model
        raise NotImplementedError

    def retrieve(
        self, query: str, corpus: list[str] | None = None, k: int = 5
    ) -> tuple[list[str], list[float]]:
        """Return top-k passages for *query*.

        Args:
            query: Query string (will be prefixed automatically).
            corpus: If provided, rebuild the index.
            k: Number of passages to return.

        Returns:
            Tuple of (passages, scores) lists.
        """
        # TODO: prepend QUERY_PREFIX to query before encoding
        raise NotImplementedError

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode *texts* into embedding vectors.

        Args:
            texts: Strings to encode.
            is_query: If True, prepend ``QUERY_PREFIX`` to each text.

        Returns:
            Shape ``(len(texts), dim)`` float32 numpy array.
        """
        # TODO: implement with BGE-specific query prefix handling
        raise NotImplementedError
