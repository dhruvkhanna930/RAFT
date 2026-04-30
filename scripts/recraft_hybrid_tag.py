#!/usr/bin/env python3
"""Re-craft hybrid Stage-2 perturbations with TAG-only invisible chars.

Background — why this script exists
-----------------------------------
The original hybrid attack used a mixed inventory of zero-width chars,
word joiners, and tag chars.  Empirical PPL profiling on GPT-2 showed:

* ZWSP, ZWJ, word joiners → +44 to +90 PPL per 5 chars (high disruption)
* Tag chars (U+E0000–U+E007F) → +11 to +18 PPL per 5 chars
* At budget=20 with TAG only, total passage PPL drops to ~11 (well below
  semantic median ~33, so PPL filters keep the hybrid and drop semantic).

This script preserves each hybrid passage's Stage-1 (semantic) text and
only re-runs Stage-2 differential evolution with the restricted TAG-only
inventory.  Stage-1 LLM crafting is skipped — saves ~80 min on 20 Qs.

Output
------
Overwrites ``results/exp07/nq/passages/hybrid_q*.json``.  Backups of the
prior (mixed-inventory) hybrid passages are saved alongside as
``hybrid_q*.json.mixed.bak``.

Usage
-----
::

    python scripts/recraft_hybrid_tag.py
    python scripts/recraft_hybrid_tag.py --budget 20 --dataset nq
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import faiss
import numpy as np
import yaml

from src.attacks.base import AttackConfig
from src.attacks.rag_pull import RAGPullAttack
from src.attacks.unicode_chars import (
    CharCategory, UnicodeInventory, perturb,
)
from src.data.index_builder import IndexBuilder
from src.data.nq_loader import NQLoader
from src.retrievers.contriever import ContrieverRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recraft_hybrid_tag")


def _load_targets(path: Path, n: int) -> list[tuple[str, list[str], str]]:
    out = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out.append((r["question"], r["gold"], r["fake"]))
    return out[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-craft hybrid passages with TAG-only inventory")
    parser.add_argument("--profile", default="small")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override perturbation budget (default: from profile)")
    args = parser.parse_args()

    cfg_path = _REPO_ROOT / "configs" / "experiments.yaml"
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    profile = raw["profiles"][args.profile]
    base = {k: v for k, v in raw.items() if k not in ("profile", "profiles")}
    base.update(profile)
    cfg = base

    n_q = cfg["n_questions"]
    n_passages = cfg["n_passages"]
    budget = args.budget if args.budget is not None else cfg["perturbation_budget"]

    src_dir = _REPO_ROOT / "results" / "exp07" / args.dataset
    pass_dir = src_dir / "passages"
    targets_path = src_dir / "targets.jsonl"
    if not targets_path.exists():
        sys.exit(f"ERROR: missing {targets_path}")

    targets = _load_targets(targets_path, n_q)
    logger.info("Loaded %d targets", len(targets))

    # ── Retriever + cached FAISS index ────────────────────────────────────────
    logger.info("Building/loading retriever and FAISS index …")
    retriever = ContrieverRetriever(device="auto", batch_size=64, normalize=True)
    data_dir = _REPO_ROOT / "data"
    loader = NQLoader(
        processed_dir=data_dir / "processed",
        corpus_size=cfg["corpus_size"],
        n_questions=n_q,
    )
    clean_corpus = loader.passages()
    builder = IndexBuilder(
        indices_dir=data_dir / "indices",
        dataset=args.dataset,
        retriever_name="contriever",
    )
    idx, clean_corpus = builder.build(clean_corpus, retriever)
    retriever.load_index(idx, clean_corpus)
    logger.info("Index ready: %d vectors", idx.ntotal)

    # ── DE attack with TAG-only inventory ─────────────────────────────────────
    tag_inventory = UnicodeInventory(categories=[CharCategory.TAG_CHARS])
    logger.info(
        "TAG-only inventory: %d chars (U+E0001..U+E007F)", len(tag_inventory),
    )
    attack_cfg = AttackConfig(injection_budget=n_passages)
    pull = RAGPullAttack(
        config=attack_cfg,
        retriever=retriever,
        char_inventory=tag_inventory,
        perturbation_budget=budget,
        de_population=cfg["de_population"],
        de_max_iter=cfg["de_max_iter"],
    )

    # ── Re-craft each question's hybrid passages ──────────────────────────────
    t_start = time.time()
    for q_idx, (question, _gold, _fake) in enumerate(targets):
        cache_file = pass_dir / f"hybrid_q{q_idx:04d}.json"
        if not cache_file.exists():
            logger.warning("[Q%02d] missing %s — skipping", q_idx, cache_file.name)
            continue

        # Backup the mixed-inventory file (one-time)
        backup = cache_file.with_suffix(".json.mixed.bak")
        if not backup.exists():
            shutil.copy2(cache_file, backup)

        # Load current hybrid passages, strip ALL invisibles to recover Stage-1 text
        with cache_file.open() as f:
            current_hybrid = json.load(f)
        semantic_texts = [
            UnicodeInventory.strip_invisible(p) for p in current_hybrid
        ]

        # Re-run Stage-2 DE on the first passage, apply same insertions to all n
        first = semantic_texts[0]
        # Trigger region: first 50 chars (matches HybridAttack default "prefix")
        end = min(50, len(first))
        valid_positions = list(range(0, end + 1))

        query_emb = retriever.encode_query(question)
        t0 = time.time()
        _, insertions = pull._optimize_with_insertions(
            first, query_emb, valid_positions,
        )
        elapsed = time.time() - t0

        new_hybrid = [perturb(t, insertions) for t in semantic_texts]
        with cache_file.open("w") as f:
            json.dump(new_hybrid, f)

        logger.info(
            "[Q%02d] DE done in %.1fs  insertions=%d  question=%r",
            q_idx, elapsed, len(insertions), question[:50],
        )

    total = time.time() - t_start
    logger.info("All %d questions re-crafted in %.1f min", len(targets), total / 60)
    logger.info("Backups at: %s/hybrid_q*.json.mixed.bak", pass_dir)


if __name__ == "__main__":
    main()
