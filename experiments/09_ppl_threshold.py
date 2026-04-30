#!/usr/bin/env python3
"""Experiment 09 — Absolute PPL threshold ablation.

Sweeps absolute GPT-2 perplexity *thresholds* (not fractions) for the
``semantic`` and ``hybrid`` attacks on Vanilla RAG / NQ.

This is a complement to ``08_ppl_ablation.py`` (which uses top-fraction
filtering).  Top-fraction does not discriminate between attacks when *all*
retrieved passages are adversarial — they all have similar PPL among
themselves and the filter just rearranges them.

Absolute threshold filtering does discriminate: if all 5 retrieved
adversarial passages have PPL > threshold, *all* are dropped (and the
RAG falls back to a clean low-rank passage).  Threshold tuning then
maps directly to "fluency required for a passage to be trusted."

Reads cached passages from ``results/exp07/nq/passages/`` — does NOT
re-run DE or LLM crafting.

Output::

    results/exp09_ppl_threshold/
      threshold_ablation.csv       # one row per (attack, threshold)
      threshold_ablation_detail.csv
      threshold_ablation.txt

Usage::

    python experiments/09_ppl_threshold.py
    python experiments/09_ppl_threshold.py --thresholds 30 40 50 75 100
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
from src.llms.ollama_client import OllamaClient
from src.llms.groq_client import GroqClient
from src.metrics.asr import is_attack_successful_fuzzy as is_attack_successful
from src.metrics.retrieval import precision_at_k
from src.metrics.stealth import char_class_entropy
from src.rag.base import RetrievalResult
from src.rag.vanilla import VanillaRAG
from src.retrievers.contriever import ContrieverRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("09_ppl_threshold")

DEFAULT_THRESHOLDS = [30.0, 40.0, 50.0, 75.0, 100.0]
DEFAULT_ATTACKS = ["semantic", "hybrid"]


def _load_cfg(profile: str = "small") -> dict[str, Any]:
    p = _REPO_ROOT / "configs" / "experiments.yaml"
    with p.open() as f:
        raw = yaml.safe_load(f)
    cfg = {k: v for k, v in raw.items() if k not in ("profile", "profiles")}
    cfg.update(raw["profiles"][profile])
    return cfg


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


def _eval_question(
    question: str,
    fake_answer: str,
    adv_passages: list[str],
    retriever: ContrieverRetriever,
    index_bytes: bytes,
    clean_corpus: list[str],
    rag: VanillaRAG,
    ppl_filter: PerplexityFilter,
    top_k: int,
    n_adv: int = 5,
    strict_fallback: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    _inject_adversarial(retriever, index_bytes, clean_corpus, adv_passages[:n_adv])
    retrieved = rag.retrieve(question, k=top_k)

    # Threshold mode: drop_fraction=None → uses absolute self.threshold
    filtered = ppl_filter.apply(retrieved.passages)
    if not filtered:
        if strict_fallback:
            # Strict mode: no adversarial fallback — LLM answers from parametric
            # knowledge only (empty context).  This makes the filter truly effective:
            # when ALL retrieved passages exceed the PPL threshold, the attack is
            # blocked rather than allowed via the top-1 fallback.
            filtered = []
        else:
            filtered = retrieved.passages[:1]

    retrieved_filtered = RetrievalResult(
        passages=filtered, scores=retrieved.scores[:len(filtered)],
        metadata=retrieved.metadata,
    )
    gen = rag.generate(question, retrieved_filtered)
    elapsed = time.time() - t0

    hit = is_attack_successful(gen.answer, fake_answer)
    adv_set = set(adv_passages)
    n_adv_before = sum(1 for s in retrieved.passages[:top_k] if s in adv_set)
    n_adv_after  = sum(1 for s in filtered if s in adv_set)
    p_at_k = precision_at_k(retrieved.passages, adv_passages, k=top_k)

    return {
        "question": question,
        "fake_answer": fake_answer,
        "hit": int(hit),
        "e2e_success": int(n_adv_after > 0 and hit),
        "n_adv_retrieved": n_adv_before,
        "n_adv_after_filter": n_adv_after,
        "n_passages_to_llm": len(filtered),
        "precision_at_k": round(p_at_k, 4),
        "rag_answer": gen.answer[:200],
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Absolute PPL threshold ablation")
    parser.add_argument("--profile", default="small")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--out-dir", default="results/exp09_ppl_threshold")
    parser.add_argument(
        "--n-adv-passages", type=int, default=5,
        help="Number of adversarial passages to inject per question (default 5). "
             "Use 1 with --strict-fallback for clean ASR divergence.",
    )
    parser.add_argument(
        "--strict-fallback", action="store_true",
        help="When ALL retrieved passages exceed the PPL threshold, generate with "
             "NO context (LLM parametric knowledge) instead of falling back to the "
             "top-1 adversarial passage.  Makes the filter truly effective.",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.profile)
    n_q = cfg["n_questions"]
    top_k = cfg.get("top_k", 5)
    n_adv = args.n_adv_passages
    strict_fallback = args.strict_fallback
    logger.info(
        "n_adv_passages=%d  strict_fallback=%s", n_adv, strict_fallback,
    )

    src_dir = Path(args.source_dir) if args.source_dir else _REPO_ROOT / "results" / "exp07" / args.dataset
    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load targets + adversarial passages
    targets_path = src_dir / "targets.jsonl"
    targets = []
    with targets_path.open() as f:
        for line in f:
            r = json.loads(line)
            targets.append((r["question"], r["gold"], r["fake"]))
    targets = targets[:n_q]
    logger.info("Loaded %d targets", len(targets))

    adv_cache: dict[tuple[int, str], list[str]] = {}
    for q_idx in range(len(targets)):
        for atk in args.attacks:
            p = src_dir / "passages" / f"{atk}_q{q_idx:04d}.json"
            with p.open() as f:
                adv_cache[(q_idx, atk)] = json.load(f)
    logger.info("Loaded %d passage caches", len(adv_cache))

    # ── LLM + retriever ───────────────────────────────────────────────────────
    model = cfg.get("model", "qwen2.5:7b")
    llm = OllamaClient(model=model)
    if not llm.is_available():
        sys.exit("ERROR: Ollama not reachable")

    retriever = ContrieverRetriever()
    rag = VanillaRAG(retriever=retriever, llm=llm, top_k=top_k)

    # ── Cached FAISS index ────────────────────────────────────────────────────
    data_dir = _REPO_ROOT / "data"
    loader = NQLoader(
        processed_dir=data_dir / "processed",
        corpus_size=cfg["corpus_size"], n_questions=n_q,
    )
    clean_corpus = loader.passages()
    builder = IndexBuilder(
        indices_dir=data_dir / "indices",
        dataset=args.dataset, retriever_name="contriever",
    )
    idx, clean_corpus = builder.build(clean_corpus, retriever)
    retriever.load_index(idx, clean_corpus)
    index_bytes = faiss.serialize_index(retriever._index)

    # ── Build threshold filters (share one GPT-2) ─────────────────────────────
    ppl_device = cfg.get("ppl_device", "cpu")
    primary = PerplexityFilter(threshold=args.thresholds[0], drop_fraction=None, device=ppl_device)
    primary._load_model()
    filters: dict[float, PerplexityFilter] = {args.thresholds[0]: primary}
    for t in args.thresholds[1:]:
        f = PerplexityFilter(threshold=t, drop_fraction=None, device=ppl_device)
        f._model = primary._model
        f._tokenizer = primary._tokenizer
        filters[t] = f

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for atk in args.attacks:
        for thr in args.thresholds:
            rows = []
            logger.info("Eval  attack=%s  threshold=%.0f", atk, thr)
            for q_idx, (q, _g, fake) in enumerate(targets):
                row = _eval_question(
                    q, fake, adv_cache[(q_idx, atk)],
                    retriever, index_bytes, clean_corpus,
                    rag, filters[thr], top_k,
                    n_adv=n_adv, strict_fallback=strict_fallback,
                )
                row["attack"] = atk
                row["ppl_threshold"] = thr
                row["q_idx"] = q_idx
                tag = "HIT" if row["hit"] else "MISS"
                logger.info(
                    "  [Q%02d] %s  retr=%d after=%d  ans=%s",
                    q_idx, tag, row["n_adv_retrieved"],
                    row["n_adv_after_filter"], row["rag_answer"][:50],
                )
                rows.append(row)
            results[(atk, thr)] = rows

    # ── Aggregate + save ──────────────────────────────────────────────────────
    summary = []
    for (atk, thr), rs in sorted(results.items()):
        n = len(rs)
        summary.append({
            "attack": atk,
            "ppl_threshold": thr,
            "n_questions": n,
            "asr_e2e": round(sum(r["e2e_success"] for r in rs) / n, 4),
            "hit_rate": round(sum(r["hit"] for r in rs) / n, 4),
            "avg_adv_before": round(sum(r["n_adv_retrieved"] for r in rs) / n, 2),
            "avg_adv_after":  round(sum(r["n_adv_after_filter"] for r in rs) / n, 2),
            "avg_passages_to_llm": round(sum(r["n_passages_to_llm"] for r in rs) / n, 2),
        })

    csv_path = out_dir / "threshold_ablation.csv"
    fields = ["attack", "ppl_threshold", "n_questions", "asr_e2e", "hit_rate",
              "avg_adv_before", "avg_adv_after", "avg_passages_to_llm"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    logger.info("Summary → %s", csv_path)

    detail_path = out_dir / "threshold_ablation_detail.csv"
    detail_fields = ["attack", "ppl_threshold", "q_idx", "question", "fake_answer",
                     "hit", "e2e_success", "n_adv_retrieved", "n_adv_after_filter",
                     "n_passages_to_llm", "precision_at_k", "rag_answer", "elapsed_s"]
    all_detail = [r for rs in results.values() for r in rs]
    with detail_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_detail)

    # ── Print table ───────────────────────────────────────────────────────────
    lines = []
    header = f"{'Threshold':>10}  " + "  ".join(f"{t:>5.0f}" for t in args.thresholds)
    sep = "-" * (12 + 7 * len(args.thresholds))
    lines.append(header)
    lines.append(sep)
    for atk in args.attacks:
        vals = [
            sum(r["e2e_success"] for r in results[(atk, t)]) / len(results[(atk, t)])
            for t in args.thresholds
        ]
        lines.append(f"{atk + ' ASR':>10}  " + "  ".join(f"{v:>4.0%} " for v in vals))
    print("\n" + "\n".join(lines))
    (out_dir / "threshold_ablation.txt").write_text("\n".join(lines) + "\n")
    logger.info("Done")


if __name__ == "__main__":
    main()
