"""Retrieval metrics: Precision/Recall/F1@k and Mean Rank.

These measure how well the attack places adversarial passages into the
retrieved context, independent of the LLM's generation.

- **P@k**: fraction of top-k retrieved passages that are adversarial.
- **R@k**: fraction of injected adversarial passages appearing in top-k.
- **F1@k**: harmonic mean of P@k and R@k.
- **Mean Rank**: average rank of the best adversarial passage across questions.
"""

from __future__ import annotations


def compute_retrieval_metrics(
    retrieved_passages: list[list[str]],
    adversarial_passages: list[list[str]],
    k: int = 5,
) -> dict[str, float]:
    """Compute retrieval metrics for a batch of queries.

    Args:
        retrieved_passages: For each query, the list of retrieved passage
            strings in rank order (index 0 = rank 1).
        adversarial_passages: For each query, the list of injected adversarial
            passage strings.
        k: Retrieval cutoff rank.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``, ``mean_rank``.
        ``mean_rank`` is the macro-average of per-query best-adversarial-rank
        (None queries — where no adversarial passage was retrieved at all —
        are excluded from the average; if all are None, returns ``float('inf')``).
    """
    if not retrieved_passages:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_rank": float("inf")}

    precisions, recalls, f1s, ranks = [], [], [], []
    for ret, adv in zip(retrieved_passages, adversarial_passages):
        p = precision_at_k(ret, adv, k)
        r = recall_at_k(ret, adv, k)
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        mr = mean_rank(ret, adv)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        if mr is not None:
            ranks.append(mr)

    avg_rank = sum(ranks) / len(ranks) if ranks else float("inf")
    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "mean_rank": avg_rank,
    }


def precision_at_k(
    retrieved: list[str], adversarial: list[str], k: int
) -> float:
    """Compute P@k for a single query.

    Args:
        retrieved: Retrieved passage list (in rank order).
        adversarial: Injected adversarial passages.
        k: Cutoff.

    Returns:
        Precision@k float in [0, 1].
    """
    if k == 0:
        return 0.0
    adv_set = set(adversarial)
    hits = sum(1 for p in retrieved[:k] if p in adv_set)
    return hits / k


def recall_at_k(
    retrieved: list[str], adversarial: list[str], k: int
) -> float:
    """Compute R@k for a single query.

    Args:
        retrieved: Retrieved passage list (in rank order).
        adversarial: Injected adversarial passages.
        k: Cutoff.

    Returns:
        Recall@k float in [0, 1].
    """
    if not adversarial:
        return 0.0
    adv_set = set(adversarial)
    hits = sum(1 for p in retrieved[:k] if p in adv_set)
    return hits / len(adversarial)


def f1_at_k(
    retrieved: list[str], adversarial: list[str], k: int
) -> float:
    """Compute F1@k for a single query (harmonic mean of P@k and R@k).

    Args:
        retrieved: Retrieved passage list (in rank order).
        adversarial: Injected adversarial passages.
        k: Cutoff.

    Returns:
        F1@k float in [0, 1].
    """
    p = precision_at_k(retrieved, adversarial, k)
    r = recall_at_k(retrieved, adversarial, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def mean_rank(
    retrieved: list[str], adversarial: list[str]
) -> float | None:
    """Return the 1-indexed rank of the first adversarial passage in *retrieved*.

    Args:
        retrieved: Full ranked retrieved list (index 0 = rank 1).
        adversarial: Injected adversarial passages.

    Returns:
        1-indexed rank of the best-ranked adversarial passage, or ``None``
        if no adversarial passage appears in *retrieved*.
    """
    adv_set = set(adversarial)
    for i, p in enumerate(retrieved):
        if p in adv_set:
            return float(i + 1)
    return None
