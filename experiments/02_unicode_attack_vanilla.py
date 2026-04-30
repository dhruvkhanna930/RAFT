"""Experiment 02 — Unicode (RAG-Pull) attack on Vanilla RAG.

Goal: replicate RAG-Pull's invisible-character retrieval boost on Vanilla RAG
with Contriever, then compare ASR and Mean Rank against Experiment 01.

Usage:
    python experiments/02_unicode_attack_vanilla.py \
        --config configs/ \
        --dataset nq \
        --n_questions 100 \
        --output results/02_unicode_vanilla_nq.json
"""

from __future__ import annotations

import argparse

# TODO: implement experiment script
#   1. Same setup as 01 but use RAGPullAttack instead of PoisonedRAGAttack
#   2. Run both raw embedding-based attack and combined stealth metrics
#   3. Compare results with 01 (ASR, Mean Rank, PPL, visual_diff_rate)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/")
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--output", default="results/02_unicode_vanilla_nq.json")
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 02."""
    args = parse_args()
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
