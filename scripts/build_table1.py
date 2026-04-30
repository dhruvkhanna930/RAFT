#!/usr/bin/env python3
"""Build Table 1: RAG variant x Attack ASR matrix (Markdown).

Reads results/exp07/summary.csv and pivots into a table where rows are
RAG variants and columns are attack types.  Default filter: defense="none"
(no defense applied) to show raw attack effectiveness.

Usage::

    python scripts/build_table1.py
    python scripts/build_table1.py --defense none --dataset nq
    python scripts/build_table1.py --summary results/exp07/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(
        description="Table 1: RAG variant x Attack ASR matrix",
    )
    p.add_argument(
        "--summary",
        default=str(_REPO_ROOT / "results" / "exp07" / "summary.csv"),
        help="Path to summary.csv",
    )
    p.add_argument("--dataset", default=None, help="Filter to this dataset (default: all)")
    p.add_argument("--defense", default="none", help="Defense condition (default: none)")
    args = p.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        sys.exit(f"ERROR: summary not found at {summary_path}")

    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit("ERROR: summary.csv is empty")

    # Filter.
    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]
    rows = [r for r in rows if r["defense"] == args.defense]

    if not rows:
        sys.exit(f"No rows match dataset={args.dataset}, defense={args.defense}")

    # Collect unique RAG variants and attacks in a stable order.
    rag_order = ["vanilla", "self_rag", "crag", "trust_rag", "robust_rag"]
    attack_order = ["semantic", "unicode", "hybrid"]

    rags_present = [r for r in rag_order if any(row["rag"] == r for row in rows)]
    attacks_present = [a for a in attack_order if any(row["attack"] == a for row in rows)]
    # Add any extras not in the predefined order.
    for r in rows:
        if r["rag"] not in rags_present:
            rags_present.append(r["rag"])
        if r["attack"] not in attacks_present:
            attacks_present.append(r["attack"])

    # Build lookup: (rag, attack) -> asr.
    # If multiple datasets, average across them.
    from collections import defaultdict
    asr_accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        asr_accum[(r["rag"], r["attack"])].append(float(r["asr"]))
    grid: dict[tuple[str, str], str] = {}
    for key, vals in asr_accum.items():
        avg = sum(vals) / len(vals)
        grid[key] = f"{avg:.0%}"

    # Print Markdown table.
    datasets_label = args.dataset or "all"
    print(f"\n## Table 1 — Attack Success Rate by RAG Variant")
    print(f"Defense: `{args.defense}` | Dataset: `{datasets_label}`\n")

    # Header.
    col_w = max(10, max(len(a) for a in attacks_present) + 2)
    header = f"| {'RAG Variant':<14} |"
    sep = f"|{'-' * 15}:|"
    for atk in attacks_present:
        header += f" {atk:^{col_w}} |"
        sep += f"{'-' * (col_w + 1)}:|"
    print(header)
    print(sep)

    # Rows.
    for rag in rags_present:
        line = f"| {rag:<14} |"
        for atk in attacks_present:
            val = grid.get((rag, atk), "---")
            line += f" {val:^{col_w}} |"
        print(line)

    print()


if __name__ == "__main__":
    main()
