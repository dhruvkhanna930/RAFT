"""Efficiency metrics.

Tracks the computational cost of generating adversarial passages:
- LLM queries per adversarial passage
- Runtime per attack
- Number of perturbations (invisible chars) needed for success
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class EfficiencyRecord:
    """Per-attack-run efficiency counters.

    Args:
        llm_queries: Total LLM API calls made during passage generation.
        runtime_seconds: Wall-clock time for the attack run.
        perturbation_count: Number of invisible chars inserted into the passage.
        iterations: Optimisation iterations run (DE generations or loop steps).
    """

    llm_queries: int = 0
    runtime_seconds: float = 0.0
    perturbation_count: int = 0
    iterations: int = 0


def compute_efficiency_metrics(
    records: list[EfficiencyRecord],
) -> dict[str, float]:
    """Aggregate efficiency records across multiple attack runs.

    Args:
        records: List of per-run ``EfficiencyRecord`` instances.

    Returns:
        Dict with keys: ``avg_llm_queries``, ``avg_runtime_s``,
        ``avg_perturbation_count``, ``avg_iterations``.
        Returns all-zero dict for empty *records*.
    """
    if not records:
        return {
            "avg_llm_queries": 0.0,
            "avg_runtime_s": 0.0,
            "avg_perturbation_count": 0.0,
            "avg_iterations": 0.0,
        }
    n = len(records)
    return {
        "avg_llm_queries": sum(r.llm_queries for r in records) / n,
        "avg_runtime_s": sum(r.runtime_seconds for r in records) / n,
        "avg_perturbation_count": sum(r.perturbation_count for r in records) / n,
        "avg_iterations": sum(r.iterations for r in records) / n,
    }


@contextmanager
def timed_attack() -> Generator[EfficiencyRecord, None, None]:
    """Context manager that records wall-clock time for an attack block.

    Usage::

        with timed_attack() as rec:
            passage = attack.generate_adversarial_passage(...)
            rec.perturbation_count = count_invisible_chars(passage.text)

    Yields:
        An ``EfficiencyRecord`` whose ``runtime_seconds`` is set on exit.
    """
    record = EfficiencyRecord()
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.runtime_seconds = time.perf_counter() - start
