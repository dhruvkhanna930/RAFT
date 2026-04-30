#!/usr/bin/env python3
"""Experiment 08 — Perplexity-filter fraction ablation.

Sweeps ``ppl_drop_fraction`` in {0.3, 0.5, 0.7, 0.9} for the
``semantic`` and ``hybrid`` attacks on Vanilla RAG / NQ.

**No re-crafting**: reuses passage caches from experiment 07
(``results/exp07/nq/passages/``).  Only the perplexity filter fraction
changes between runs.

Goal: show that as the PPL filter becomes more aggressive, semantic ASR
collapses while hybrid ASR stays flat — because hybrid passages are
natural prose (low PPL) that survives stricter filtering.

Output::

    results/exp08_ppl_ablation/
      ppl_ablation.csv      # one row per (attack, ppl_fraction)
      ppl_ablation.txt      # human-readable table (printed to stdout too)

Usage::

    python experiments/08_ppl_ablation.py
    python experiments/08_ppl_ablation.py --ppl-fractions 0.3 0.5 0.7 0.9
    python experiments/08_ppl_ablation.py --profile small --dataset nq
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import faiss
import numpy as np
import yaml

from src.attacks.unicode_chars import UnicodeInventory
from src.data.index_builder import IndexBuilder
from src.data.nq_loader import NQLoader
from src.defenses.perplexity import PerplexityFilter
from src.defenses.unicode_normalize import UnicodeNormalizer
from src.defenses.zero_width_strip import ZeroWidthStripDefense
from src.llms.ollama_client import OllamaClient
from src.llms.groq_client import GroqClient
from src.metrics.asr import is_attack_successful_fuzzy as is_attack_successful
from src.metrics.retrieval import precision_at_k, recall_at_k
from src.metrics.stealth import char_class_entropy, visual_diff_rate
from src.rag.base import RetrievalResult
from src.rag.vanilla import VanillaRAG
from src.retrievers.contriever import ContrieverRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("08_ppl_ablation")

DEFAULT_PPL_FRACTIONS = [0.3, 0.5, 0.7, 0.9]
DEFAULT_ATTACKS = ["semantic", "hybrid"]


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(profile: str = "small") -> dict[str, Any]:
    cfg_path = _REPO_ROOT / "configs" / "experiments.yaml"
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    profile_data = raw.get("profiles", {}).get(profile, {})
    base = {k: v for k, v in raw.items() if k not in ("profiles", "profile")}
    base.update(profile_data)
    return base


# ── Index helpers (identical to 07_full_matrix) ───────────────────────────────

def _inject_adversarial(
    retriever: ContrieverRetriever,
    index_bytes: bytes,
    clean_corpus: list[str],
    adv_passages: list[str],
) -> None:
    adv_emb = retriever.encode_passages(adv_passages, batch_size=64).astype(np.float32)
    faiss.normalize_L2(adv_emb)
    idx = faiss.deserialize_index(index_bytes)
    idx.add(adv_emb)
    retriever.load_index(idx, clean_corpus + adv_passages)


# ── Per-question evaluation ───────────────────────────────────────────────────

def _eval_question(
    question: str,
    fake_answer: str,
    adv_passages_raw: list[str],
    retriever: ContrieverRetriever,
    index_bytes: bytes,
    clean_corpus: list[str],
    rag: VanillaRAG,
    ppl_filter: PerplexityFilter,
    top_k: int,
) -> dict[str, Any]:
    t0 = time.time()

    # Adversarial passages injected as-is (raw — no pre-index normalization here)
    _inject_adversarial(retriever, index_bytes, clean_corpus, adv_passages_raw)
    retrieved = rag.retrieve(question, k=top_k)

    # Apply perplexity filter (post-retrieval)
    filtered = ppl_filter.apply(retrieved.passages)
    if not filtered:
        filtered = retrieved.passages[:1]
    retrieved_filtered = RetrievalResult(
        passages=filtered,
        scores=retrieved.scores[:len(filtered)],
        metadata=retrieved.metadata,
    )

    gen_result = rag.generate(question, retrieved_filtered)
    elapsed = time.time() - t0

    hit = is_attack_successful(gen_result.answer, fake_answer)

    # Retrieval metrics: compare against the (raw) adversarial set
    adv_set = set(adv_passages_raw)
    n_adv_before = sum(1 for s in retrieved.passages[:top_k] if s in adv_set)
    n_adv_after  = sum(1 for s in filtered if s in adv_set)
    p = precision_at_k(retrieved.passages, adv_passages_raw, k=top_k)

    zw_fracs = [char_class_entropy(ap)["zerowidth_frac"] for ap in adv_passages_raw]
    avg_zw = sum(zw_fracs) / len(zw_fracs) if zw_fracs else 0.0

    return {
        "question": question,
        "fake_answer": fake_answer,
        "hit": int(hit),
        "e2e_success": int(n_adv_after > 0 and hit),
        "n_adv_retrieved": n_adv_before,
        "n_adv_after_filter": n_adv_after,
        "precision_at_k": round(p, 4),
        "n_passages_to_llm": len(filtered),
        "avg_zw_frac": round(avg_zw, 6),
        "rag_answer": gen_result.answer[:200],
        "elapsed_s": round(elapsed, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PPL-fraction ablation")
    parser.add_argument("--profile", default="small")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument(
        "--ppl-fractions", nargs="+", type=float, default=DEFAULT_PPL_FRACTIONS,
        metavar="F", help="PPL drop fractions to sweep",
    )
    parser.add_argument(
        "--attacks", nargs="+", default=DEFAULT_ATTACKS,
        help="Attacks to evaluate (default: semantic hybrid)",
    )
    parser.add_argument(
        "--source-dir", default=None,
        help="Path to exp07 output dir (default: results/exp07/{dataset})",
    )
    parser.add_argument("--out-dir", default="results/exp08_ppl_ablation")
    args = parser.parse_args()

    cfg = _load_config(args.profile)
    ppl_fractions: list[float] = sorted(args.ppl_fractions)
    attacks: list[str] = args.attacks

    source_dir = Path(args.source_dir or (_REPO_ROOT / "results" / "exp07" / args.dataset))
    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    passage_cache_dir = source_dir / "passages"
    targets_path = source_dir / "targets.jsonl"

    if not targets_path.exists():
        sys.exit(f"ERROR: targets file not found at {targets_path}\n"
                 "Run experiment 07 first to generate passage caches.")

    # ── Load targets ──────────────────────────────────────────────────────────
    targets: list[tuple[str, list[str], str]] = []
    with targets_path.open() as f:
        for line in f:
            rec = json.loads(line)
            targets.append((rec["question"], rec["gold"], rec["fake"]))
    n_q = cfg.get("n_questions", 20)
    targets = targets[:n_q]
    logger.info("Loaded %d targets from %s", len(targets), targets_path)

    # ── Load passage caches ───────────────────────────────────────────────────
    adv_cache: dict[tuple[int, str], list[str]] = {}
    missing = []
    for q_idx in range(len(targets)):
        for atk in attacks:
            p = passage_cache_dir / f"{atk}_q{q_idx:04d}.json"
            if p.exists():
                with p.open() as f:
                    adv_cache[(q_idx, atk)] = json.load(f)
            else:
                missing.append(str(p))
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} passage cache files missing.\n"
            f"First missing: {missing[0]}\n"
            "Run experiment 07 to generate them."
        )
    logger.info("Loaded passage caches for %d (question, attack) pairs", len(adv_cache))

    # ── LLM + Retriever ───────────────────────────────────────────────────────
    model = cfg.get("model", "qwen2.5:7b")
    llm_backend = cfg.get("llm_backend", "ollama")
    if llm_backend == "groq":
        llm = GroqClient(model=model)
    else:
        llm = OllamaClient(model=model)
        if not llm.is_available():
            sys.exit("ERROR: Ollama not reachable — run: ollama serve")

    retriever = ContrieverRetriever()
    rag = VanillaRAG(retriever=retriever, llm=llm, top_k=cfg.get("top_k", 5))
    top_k = cfg.get("top_k", 5)

    # ── Corpus + index (use cached FAISS index from exp07 if present) ─────────
    logger.info("Loading corpus …")
    data_dir = _REPO_ROOT / "data"
    corpus_size = cfg.get("corpus_size", 10000)
    loader = NQLoader(
        processed_dir=data_dir / "processed",
        corpus_size=corpus_size,
        n_questions=n_q,
    )
    clean_corpus = loader.passages()
    logger.info("Corpus: %d passages", len(clean_corpus))

    logger.info("Loading FAISS index (cached if present)…")
    builder = IndexBuilder(
        indices_dir=data_dir / "indices",
        dataset=args.dataset,
        retriever_name="contriever",
    )
    t0 = time.time()
    idx, clean_corpus = builder.build(clean_corpus, retriever)
    retriever.load_index(idx, clean_corpus)
    index_bytes = faiss.serialize_index(retriever._index)
    logger.info(
        "Index ready  vectors=%d  (%.1fs)",
        idx.ntotal, time.time() - t0,
    )

    # ── Build PPL filters: one GPT-2 model shared across all fractions ────────
    # Loading GPT-2 four times would OOM the Mac (each ~500MB + Contriever +
    # Ollama qwen2.5:7b ~5GB + 10k corpus embeddings).  Instantiate one filter,
    # warm-load its GPT-2, then share the model/tokenizer with the others.
    ppl_device = cfg.get("ppl_device", "cpu")
    logger.info("Loading shared PPL scorer (GPT-2) on %s …", ppl_device)
    primary = PerplexityFilter(drop_fraction=ppl_fractions[0], device=ppl_device)
    primary._load_model()
    logger.info("GPT-2 loaded; building %d filters that share the model", len(ppl_fractions))

    ppl_filters: dict[float, PerplexityFilter] = {ppl_fractions[0]: primary}
    for frac in ppl_fractions[1:]:
        f = PerplexityFilter(drop_fraction=frac, device=ppl_device)
        f._model = primary._model
        f._tokenizer = primary._tokenizer
        ppl_filters[frac] = f

    # ── Sweep ─────────────────────────────────────────────────────────────────
    # Results: {(attack, ppl_fraction): list[dict]}
    results: dict[tuple[str, float], list[dict[str, Any]]] = {}

    for atk in attacks:
        for frac in ppl_fractions:
            cell_key = (atk, frac)
            rows: list[dict[str, Any]] = []
            logger.info(
                "Evaluating  attack=%s  ppl_fraction=%.1f  n=%d questions",
                atk, frac, len(targets),
            )
            for q_idx, (question, gold, fake) in enumerate(targets):
                adv = adv_cache[(q_idx, atk)]
                row = _eval_question(
                    question, fake, adv,
                    retriever, index_bytes, clean_corpus,
                    rag, ppl_filters[frac], top_k,
                )
                row["attack"] = atk
                row["ppl_fraction"] = frac
                row["q_idx"] = q_idx
                tag = "HIT" if row["hit"] else "MISS"
                logger.info(
                    "  [Q%02d] %s  adv_retrieved=%d  adv_after_filter=%d  ans=%s",
                    q_idx, tag, row["n_adv_retrieved"], row["n_adv_after_filter"],
                    row["rag_answer"][:50],
                )
                rows.append(row)
            results[cell_key] = rows

    # ── Aggregate + save ──────────────────────────────────────────────────────
    summary_rows: list[dict[str, Any]] = []
    for (atk, frac), rows in sorted(results.items()):
        n = len(rows)
        asr      = sum(r["e2e_success"] for r in rows) / n
        hit_rate = sum(r["hit"] for r in rows) / n
        avg_adv_before = sum(r["n_adv_retrieved"] for r in rows) / n
        avg_adv_after  = sum(r["n_adv_after_filter"] for r in rows) / n
        avg_pk   = sum(r["precision_at_k"] for r in rows) / n
        summary_rows.append({
            "attack": atk,
            "ppl_fraction": frac,
            "n_questions": n,
            "asr_e2e": round(asr, 4),
            "hit_rate": round(hit_rate, 4),
            "avg_adv_before_filter": round(avg_adv_before, 2),
            "avg_adv_after_filter": round(avg_adv_after, 2),
            "mean_p_at_k": round(avg_pk, 4),
        })

    csv_path = out_dir / "ppl_ablation.csv"
    fields = [
        "attack", "ppl_fraction", "n_questions",
        "asr_e2e", "hit_rate",
        "avg_adv_before_filter", "avg_adv_after_filter",
        "mean_p_at_k",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Summary CSV → %s", csv_path)

    # Also dump per-question detail
    detail_path = out_dir / "ppl_ablation_detail.csv"
    detail_fields = [
        "attack", "ppl_fraction", "q_idx", "question", "fake_answer",
        "hit", "e2e_success",
        "n_adv_retrieved", "n_adv_after_filter",
        "precision_at_k", "n_passages_to_llm",
        "avg_zw_frac", "rag_answer", "elapsed_s",
    ]
    all_detail = [r for rows in results.values() for r in rows]
    with detail_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_detail)
    logger.info("Detail CSV → %s", detail_path)

    # ── Print table ───────────────────────────────────────────────────────────
    table_lines = []
    header = f"\n{'PPL drop fraction':>20}  " + "  ".join(f"{f:.1f}" for f in ppl_fractions)
    sep    = "-" * (22 + 6 * len(ppl_fractions))
    table_lines.append(header)
    table_lines.append(sep)

    for atk in attacks:
        asrs = [
            results[(atk, f)] for f in ppl_fractions if (atk, f) in results
        ]
        asr_vals = [
            sum(r["e2e_success"] for r in rows) / len(rows)
            for rows in asrs
        ]
        row_str = f"  {'ASR  ' + atk:>18}  " + "  ".join(f"{v:.0%}" for v in asr_vals)
        table_lines.append(row_str)

    # Also show hit_rate (LLM reproduced fake regardless of retrieval score)
    table_lines.append(sep)
    for atk in attacks:
        vals = [
            sum(r["hit"] for r in results[(atk, f)]) / len(results[(atk, f)])
            for f in ppl_fractions if (atk, f) in results
        ]
        row_str = f"  {'Hit  ' + atk:>18}  " + "  ".join(f"{v:.0%}" for v in vals)
        table_lines.append(row_str)

    table_lines.append(sep)
    table_str = "\n".join(table_lines)
    print(table_str)

    txt_path = out_dir / "ppl_ablation.txt"
    txt_path.write_text(table_str + "\n")
    logger.info("Text table → %s", txt_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
