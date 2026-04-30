"""Experiment 04 — Main matrix: all attacks × all RAG variants × all datasets.

This is the headline experiment for the paper.
Matrix: 3 attacks × 5 RAG variants × 3 datasets × 3 LLMs (final runs).

Dev mode runs one LLM (Ollama llama3.1:8b) and 100 questions per dataset.
Production mode runs all 3 LLMs and the full dev subsample.

Usage:
    # Dev run (fast)
    python experiments/04_attack_across_rag_variants.py \
        --config configs/ --mode dev \
        --output results/04_main_matrix_dev.json

    # Production run (slow, API LLMs)
    python experiments/04_attack_across_rag_variants.py \
        --config configs/ --mode prod \
        --output results/04_main_matrix_prod.json
"""

from __future__ import annotations

import argparse

# TODO: implement experiment script
#   Build nested loop: dataset × attack × rag_variant × llm
#   Use timed_attack() context manager for efficiency metrics
#   Aggregate and save full result matrix


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/")
    parser.add_argument("--mode", default="dev", choices=["dev", "prod"])
    parser.add_argument("--dataset", default=None, help="Run single dataset (default: all)")
    parser.add_argument("--output", default="results/04_main_matrix.json")
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 04."""
    args = parse_args()
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
