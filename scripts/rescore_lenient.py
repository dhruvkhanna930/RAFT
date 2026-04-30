#!/usr/bin/env python3
"""Re-score existing exp07/exp08 CSVs using the lenient (fuzzy) ASR metric.

The original strict-substring metric mis-classifies attacks as failures when
the LLM output contains invisible chars from the adversarial passage or has
minor typos (e.g. "Reena Shamshukhin" vs target "Reena Shamshukha").  The
fuzzy matcher in :mod:`src.metrics.asr` correctly counts these as hits.

Reads ``rag_answer`` + ``fake_answer`` columns from each detail CSV and
overwrites ``hit`` and ``e2e_success`` in place.  A backup of each file is
saved with a ``.strict.bak`` suffix.

Also regenerates ``results/exp07/summary.csv`` from the rescored detail CSVs.

Usage::

    python scripts/rescore_lenient.py
    python scripts/rescore_lenient.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.metrics.asr import is_attack_successful_fuzzy


def _rescore_detail_csv(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Re-score one detail CSV in place. Returns (n_rows, n_flips)."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0

    flips = 0
    for r in rows:
        if "rag_answer" not in r or "fake_answer" not in r:
            return len(rows), 0
        old_hit = r.get("hit", "0").strip()
        new_hit = "1" if is_attack_successful_fuzzy(r["rag_answer"], r["fake_answer"]) else "0"
        if old_hit != new_hit:
            flips += 1
        r["hit"] = new_hit
        # e2e requires both retrieval and hit; preserve original retrieval signal
        n_adv_key = "n_adv_in_top_k" if "n_adv_in_top_k" in r else "n_adv_after_filter"
        if n_adv_key in r:
            n_adv = int(float(r.get(n_adv_key, "0") or "0"))
            r["e2e_success"] = "1" if (n_adv > 0 and new_hit == "1") else "0"
        else:
            r["e2e_success"] = new_hit

    if dry_run:
        return len(rows), flips

    backup = path.with_suffix(path.suffix + ".strict.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows), flips


def _regenerate_exp07_summary(detail_dir: Path) -> Path:
    """Rebuild ``results/exp07/summary.csv`` from rescored detail CSVs."""
    summary_path = detail_dir.parent / "summary.csv"
    summary_rows = []
    for csv_path in sorted(detail_dir.glob("*_*_*.csv")):
        # Skip non-detail CSVs (e.g. summary.csv at this level)
        if csv_path.name == "summary.csv": continue
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows: continue
        n = len(rows)
        # Filename pattern: {rag}_{attack}_{defense}.csv  (defense may include "_")
        stem = csv_path.stem
        parts = stem.split("_")
        rag = parts[0]
        if rag == "self" or rag == "trust" or rag == "robust":
            rag = parts[0] + "_" + parts[1]
            attack = parts[2]
            defense = "_".join(parts[3:])
        else:
            attack = parts[1]
            defense = "_".join(parts[2:])
        asr      = sum(int(r["hit"])         for r in rows) / n
        e2e_mean = sum(int(r["e2e_success"]) for r in rows) / n
        p_at_k   = sum(float(r["precision_at_k"]) for r in rows) / n
        r_at_k   = sum(float(r["recall_at_k"])    for r in rows) / n
        f_at_k   = sum(float(r["f1_at_k"])        for r in rows) / n
        summary_rows.append({
            "dataset": rows[0]["dataset"],
            "rag": rag,
            "attack": attack,
            "defense": defense,
            "n_questions": n,
            "asr": round(asr, 4),
            "mean_e2e": round(e2e_mean, 4),
            "mean_p_at_k": round(p_at_k, 4),
            "mean_r_at_k": round(r_at_k, 4),
            "mean_f1_at_k": round(f_at_k, 4),
        })

    fields = [
        "dataset", "rag", "attack", "defense",
        "n_questions", "asr", "mean_e2e",
        "mean_p_at_k", "mean_r_at_k", "mean_f1_at_k",
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score CSVs with lenient ASR")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument(
        "--root", default="results", help="Results root (default: results/)",
    )
    args = parser.parse_args()

    results_root = _REPO_ROOT / args.root
    if not results_root.exists():
        sys.exit(f"ERROR: results root not found at {results_root}")

    total_rows = 0
    total_flips = 0
    files_touched = 0

    for csv_path in sorted(results_root.rglob("*.csv")):
        # Skip non-detail summary files and backups
        if csv_path.name in {"summary.csv", "ppl_ablation.csv"}:
            continue
        if csv_path.suffix == ".bak" or ".strict" in csv_path.suffixes:
            continue
        n_rows, flips = _rescore_detail_csv(csv_path, dry_run=args.dry_run)
        if n_rows == 0:
            continue
        total_rows += n_rows
        total_flips += flips
        files_touched += 1
        marker = "[DRY] " if args.dry_run else ""
        print(f"{marker}{csv_path.relative_to(_REPO_ROOT)}  rows={n_rows}  flipped→hit={flips}")

    print(f"\nTotal: {files_touched} files, {total_rows} rows, {total_flips} hit flips")

    # Regenerate exp07 summary if it exists
    exp07_dir = results_root / "exp07" / "nq"
    if exp07_dir.exists() and not args.dry_run:
        path = _regenerate_exp07_summary(exp07_dir)
        print(f"Regenerated → {path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
