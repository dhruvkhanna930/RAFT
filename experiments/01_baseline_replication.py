"""Experiment 01 — PoisonedRAG baseline replication.

Goal: reproduce PoisonedRAG Table 1 numbers on NQ with
Contriever + Llama 3.1 8B (local via Ollama).

Success criterion: ASR ≥ 90% on 100 NQ questions (matching PoisonedRAG §5).
If this passes, the pipeline is correctly wired and Phase 2 can begin.

Usage:
    python experiments/01_baseline_replication.py \
        --config configs/ \
        --dataset nq \
        --n_questions 100 \
        --output results/01_baseline_nq.json
"""

from __future__ import annotations

import argparse

# TODO: implement experiment script
#   1. Parse CLI args (config dir, dataset, n_questions, output path)
#   2. Load configs via src.utils.io.load_yaml
#   3. set_seed(42)
#   4. Load NQ dev subsample via HuggingFace datasets
#   5. Initialise ContrieverRetriever and OllamaClient
#   6. Initialise PoisonedRAGAttack from config
#   7. Initialise VanillaRAG
#   8. For each question:
#      a. Generate adversarial passages
#      b. Inject into corpus
#      c. Run VanillaRAG.answer(question, poisoned_corpus)
#      d. Record result
#   9. Compute and log ASR + retrieval metrics
#   10. Save results to output path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/", help="Config directory path")
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--output", default="results/01_baseline_nq.json")
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 01."""
    args = parse_args()
    # TODO: implement (see module docstring outline)
    raise NotImplementedError


if __name__ == "__main__":
    main()
