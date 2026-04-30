"""Experiment 03 — All three attacks vs. all four defenses on Vanilla RAG.

Goal: measure defense bypass rate for each (attack, defense) pair.
Defenses tested: paraphrase, PPL filter, NFKC normalisation, duplicate filter.

Produces the defense-bypass table (Table X in the paper).

Usage:
    python experiments/03_attack_vs_defenses.py \
        --config configs/ \
        --dataset nq \
        --n_questions 100 \
        --output results/03_attack_vs_defenses_nq.json
"""

from __future__ import annotations

import argparse

# TODO: implement experiment script
#   For each attack in [PoisonedRAG, RAGPull, Hybrid]:
#     For each defense in [None, Paraphrase, PPL, UnicodeNorm, Duplicate]:
#       Run attack on Vanilla RAG with defense applied
#       Record ASR, retrieval metrics, stealth metrics
#   Save full matrix to output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/")
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--output", default="results/03_attack_vs_defenses_nq.json")
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 03."""
    args = parse_args()
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
