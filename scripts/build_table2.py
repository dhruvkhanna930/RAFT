#!/usr/bin/env python3
"""Build Table 2: Attack x Defense bypass matrix (Markdown).

Reads results/exp07/summary.csv and pivots into a table where rows are
attack types and columns are defense conditions.  Values are ASR,
optionally averaged across RAG variants.

Usage::

    python scripts/build_table2.py
    python scripts/build_table2.py --rag vanilla --dataset nq
    python scripts/build_table2.py --summary results/exp07/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(
        description="Table 2: Attack x Defense bypass matrix",
    )
    p.add_argument(
        "--summary",
        default=str(_REPO_ROOT / "results" / "exp07" / "summary.csv"),
        help="Path to summary.csv",
    )
    p.add_argument("--dataset", default=None, help="Filter to this dataset (default: all)")
    p.add_argument("--rag", default=None, help="Filter to this RAG variant (default: average all)")
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
    if args.rag:
        rows = [r for r in rows if r["rag"] == args.rag]

    if not rows:
        sys.exit(f"No rows match dataset={args.dataset}, rag={args.rag}")

    # Collect unique attacks and defenses in a stable order.
    attack_order = ["semantic", "unicode", "hybrid"]
    defense_order = ["none", "query_paraphrase", "perplexity", "unicode_normalize", "zero_width_strip"]

    attacks_present = [a for a in attack_order if any(r["attack"] == a for r in rows)]
    defenses_present = [d for d in defense_order if any(r["defense"] == d for r in rows)]
    for r in rows:
        if r["attack"] not in attacks_present:
            attacks_present.append(r["attack"])
        if r["defense"] not in defenses_present:
            defenses_present.append(r["defense"])

    # Build lookup: (attack, defense) -> list of ASR values.
    asr_accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        asr_accum[(r["attack"], r["defense"])].append(float(r["asr"]))

    grid: dict[tuple[str, str], str] = {}
    for key, vals in asr_accum.items():
        avg = sum(vals) / len(vals)
        grid[key] = f"{avg:.0%}"

    # Print Markdown table.
    rag_label = args.rag or "avg(all)"
    datasets_label = args.dataset or "all"
    print(f"\n## Table 2 — Defense Bypass Rate by Attack")
    print(f"RAG: `{rag_label}` | Dataset: `{datasets_label}`\n")

    # Header.
    col_w = max(10, max(len(d) for d in defenses_present) + 2)
    header = f"| {'Attack':<10} |"
    sep = f"|{'-' * 11}:|"
    for d in defenses_present:
        header += f" {d:^{col_w}} |"
        sep += f"{'-' * (col_w + 1)}:|"
    print(header)
    print(sep)

    # Rows.
    for atk in attacks_present:
        line = f"| {atk:<10} |"
        for d in defenses_present:
            val = grid.get((atk, d), "---")
            line += f" {val:^{col_w}} |"
        print(line)

    print()


if __name__ == "__main__":
    main()
