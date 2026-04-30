"""Experiment 05 — Ablation studies.

Isolates the contribution of each component in the Hybrid attack:
1. Semantic-only vs. Unicode-only vs. Hybrid (main ablation)
2. Perturbation budget sensitivity (5, 10, 20, 30, 50 chars)
3. Character category ablation (which invisible char types matter most)
4. Trigger location ablation (prefix vs. suffix vs. interleaved)
5. Embedder ablation (Contriever vs. BGE)

Usage:
    python experiments/05_ablations.py \
        --config configs/ \
        --ablation perturbation_budget \
        --output results/05_ablation_budget.json
"""

from __future__ import annotations

import argparse

ABLATIONS = [
    "component",           # semantic vs. unicode vs. hybrid
    "perturbation_budget", # budget sensitivity
    "char_category",       # which char types
    "trigger_location",    # prefix/suffix/interleaved
    "embedder",            # contriever vs. bge
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/")
    parser.add_argument(
        "--ablation",
        default="component",
        choices=ABLATIONS,
        help="Which ablation to run",
    )
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--output", default="results/05_ablations.json")
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 05."""
    args = parse_args()
    # TODO: dispatch to the appropriate ablation runner based on args.ablation
    raise NotImplementedError


if __name__ == "__main__":
    main()
