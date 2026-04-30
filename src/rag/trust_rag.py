"""TrustRAG variant (Zhou et al., 2025).

Defends against PoisonedRAG-style injection by K-means clustering retrieved
passage embeddings: adversarial passages tend to form the minority cluster,
which is discarded before generation.

**Deviation from original:** None — the core algorithm is K-means on passage
embeddings with majority-cluster selection.  We use scikit-learn KMeans
(same algorithm, different implementation from the upstream repo's custom code).

Upstream repo: https://github.com/HuichiZhou/TrustRAG  (MIT, Jan 2025)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from src.rag.base import GenerationResult, RagBase, RetrievalResult

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = """\
Answer the question based on the given passages. \
Only give me the answer and do not output any other words.

The following are given passages.
{passages}

Answer the question based on the given passages. \
Only give me the answer and do not output any other words.
Question: {question}
Answer:"""


class TrustRAG(RagBase):
    """TrustRAG: K-means filtering to remove potentially poisoned passages.

    Retrieves ``retrieve_k > top_k`` candidates, embeds them, clusters into
    K groups, and keeps only the majority cluster as trusted context.

    Args:
        retriever: Base retriever (must support ``embed(texts)``).
        llm: LLM for generation.
        top_k: Final number of passages used for generation (after filtering).
        retrieve_k: Initial retrieval size before K-means filtering.
        n_clusters: K-means cluster count (default 2).
        trust_threshold: Minimum cluster-size ratio to accept a cluster.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        top_k: int = 5,
        retrieve_k: int = 20,
        n_clusters: int = 2,
        trust_threshold: float = 0.8,
    ) -> None:
        super().__init__(retriever, llm, top_k)
        self.retrieve_k = retrieve_k
        self.n_clusters = n_clusters
        self.trust_threshold = trust_threshold

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        """Retrieve then K-means filter to remove adversarial passages.

        1. Retrieve ``self.retrieve_k`` candidates via the base retriever.
        2. Embed all candidates via ``retriever.embed(passages)``.
        3. K-means cluster the embeddings.
        4. Keep only passages in the majority (largest) cluster.
        5. Truncate to top-*k* by original retrieval score.

        Args:
            query: User question.
            k: Number of passages to return after filtering.

        Returns:
            ``RetrievalResult`` with only majority-cluster passages.
        """
        passages, scores = self.retriever.retrieve(query, k=self.retrieve_k)

        # Edge case: fewer passages retrieved than clusters requested.
        if len(passages) <= self.n_clusters:
            return RetrievalResult(
                passages=passages[:k],
                scores=scores[:k],
                metadata={"kmeans_skipped": True},
            )

        embeddings = np.array(self.retriever.embed(passages))
        filtered_passages, filtered_scores, meta = self._kmeans_filter(
            embeddings, passages, scores
        )

        # Truncate to top-k by score.
        filtered_passages = filtered_passages[:k]
        filtered_scores = filtered_scores[:k]

        return RetrievalResult(
            passages=filtered_passages,
            scores=filtered_scores,
            metadata=meta,
        )

    def generate(self, query: str, retrieved: RetrievalResult) -> GenerationResult:
        """Generate answer from K-means-filtered trusted passages.

        Uses the same PoisonedRAG-style prompt template as VanillaRAG.

        Args:
            query: User question.
            retrieved: Filtered ``RetrievalResult``.

        Returns:
            ``GenerationResult`` with cluster labels in metadata.
        """
        passages_block = "\n\n".join(
            f"Passage {i + 1}: {p}"
            for i, p in enumerate(retrieved.passages)
        )
        prompt = _GENERATION_PROMPT.format(
            passages=passages_block,
            question=query,
        )
        answer = self.llm.generate(prompt)
        return GenerationResult(
            answer=answer,
            retrieved=retrieved,
            metadata={"cluster_info": retrieved.metadata},
        )

    def _kmeans_filter(
        self,
        embeddings: np.ndarray,
        passages: list[str],
        scores: list[float],
    ) -> tuple[list[str], list[float], dict[str, Any]]:
        """Run K-means and return passages from the majority cluster.

        Args:
            embeddings: Shape ``(N, dim)`` passage embedding matrix.
            passages: Corresponding passage texts.
            scores: Corresponding retrieval scores.

        Returns:
            Tuple of ``(filtered_passages, filtered_scores, cluster_metadata)``.
        """
        n_samples = embeddings.shape[0]
        effective_clusters = min(self.n_clusters, n_samples)

        km = KMeans(
            n_clusters=effective_clusters,
            random_state=42,
            n_init=10,
        )
        labels = km.fit_predict(embeddings)

        # Identify majority cluster.
        counts = Counter(int(l) for l in labels)
        majority_label = counts.most_common(1)[0][0]

        logger.debug(
            "TrustRAG K-means: %d clusters, sizes=%s, majority=%d",
            effective_clusters,
            dict(counts),
            majority_label,
        )

        # Keep passages in the majority cluster, preserving score order.
        filtered_passages: list[str] = []
        filtered_scores: list[float] = []
        for passage, score, label in zip(passages, scores, labels):
            if int(label) == majority_label:
                filtered_passages.append(passage)
                filtered_scores.append(score)

        meta: dict[str, Any] = {
            "labels": [int(l) for l in labels],
            "cluster_sizes": dict(counts),
            "majority_label": majority_label,
            "n_kept": len(filtered_passages),
            "n_discarded": n_samples - len(filtered_passages),
        }

        return filtered_passages, filtered_scores, meta
