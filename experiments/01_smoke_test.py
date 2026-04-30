#!/usr/bin/env python3
"""Smoke test — 10 NQ questions through VanillaRAG (no attack).

Verifies the full retrieve-then-generate pipeline end-to-end.
Expected runtime: < 5 minutes on M1 with qwen2.5:7b.

Usage::

    # Basic run
    python experiments/01_smoke_test.py

    # Use a different model or corpus size
    python experiments/01_smoke_test.py --model mistral:latest --corpus-size 500

Prerequisites:
    ollama serve          # start the Ollama daemon
    ollama pull qwen2.5:7b
    python scripts/build_index.py --dataset nq --corpus-size 1000  # optional, for cached index
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running directly from the repo root or the experiments/ directory.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.data.index_builder import IndexBuilder
from src.data.nq_loader import NQLoader
from src.llms.ollama_client import OllamaClient
from src.rag.vanilla import VanillaRAG
from src.retrievers.contriever import ContrieverRetriever


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag")
    p.add_argument(
        "--corpus-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of passages to index (uses cache if available)",
    )
    p.add_argument(
        "--n-questions", type=int, default=10, metavar="N", help="Questions to run"
    )
    p.add_argument("--top-k", type=int, default=5, help="Passages to retrieve per query")
    p.add_argument("--max-tokens", type=int, default=80, help="Max answer tokens")
    p.add_argument(
        "--data-dir", default=str(_REPO_ROOT / "data"), help="Root data directory"
    )
    return p.parse_args()


def _check_ollama(llm: OllamaClient) -> None:
    if not llm.is_available():
        print(
            "ERROR: Ollama is not reachable at http://localhost:11434\n"
            "Start it with:  ollama serve",
            file=sys.stderr,
        )
        sys.exit(1)

    pulled = llm.list_models()
    if llm.model not in pulled:
        print(
            f"ERROR: Model '{llm.model}' is not pulled.\n"
            f"Available models: {pulled or '(none)'}\n"
            f"Pull it with:  ollama pull {llm.model}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Ollama  OK  —  model: {llm.model}")


def _build_or_load_index(
    retriever: ContrieverRetriever,
    passages: list[str],
    data_dir: Path,
) -> None:
    """Use cached FAISS index if it exists, otherwise encode and build."""
    builder = IndexBuilder(
        indices_dir=data_dir / "indices",
        dataset="nq",
        retriever_name="contriever",
    )
    try:
        index, cached_corpus = builder.load(corpus_size=len(passages))
        retriever.load_index(index, cached_corpus)
        print(f"Index   OK  —  loaded from cache  ({len(cached_corpus)} passages)")
    except FileNotFoundError:
        print(
            f"Index   —   no cache found, encoding {len(passages)} passages "
            f"on {retriever.device}…",
            flush=True,
        )
        t0 = time.time()
        retriever.build_index(passages)
        print(f"Index   OK  —  built in {time.time() - t0:.1f}s")


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    wall_start = time.time()

    # ── 1. Ollama health check ─────────────────────────────────────────────
    llm = OllamaClient(
        model=args.model,
        temperature=0.0,   # greedy — deterministic for smoke test
        max_tokens=args.max_tokens,
    )
    _check_ollama(llm)

    # ── 2. Load NQ data ────────────────────────────────────────────────────
    loader = NQLoader(
        processed_dir=data_dir / "processed",
        corpus_size=args.corpus_size,
        n_questions=args.n_questions,
    )
    passages = loader.passages()
    questions = loader.questions()[: args.n_questions]
    print(f"Data    OK  —  {len(passages)} passages, {len(questions)} questions")

    # ── 3. Retriever + index ───────────────────────────────────────────────
    retriever = ContrieverRetriever(device="auto", batch_size=64, normalize=True)
    _build_or_load_index(retriever, passages, data_dir)

    # ── 4. VanillaRAG ─────────────────────────────────────────────────────
    rag = VanillaRAG(retriever=retriever, llm=llm, top_k=args.top_k)
    print(f"RAG     OK  —  VanillaRAG  top_k={args.top_k}")
    print()

    # ── 5. Run ────────────────────────────────────────────────────────────
    for i, q_rec in enumerate(questions, start=1):
        question = q_rec["question"]
        gold = q_rec.get("answers", [])

        print(f"{'─' * 72}")
        print(f"[{i:02d}/{len(questions)}]  {question}")

        t = time.time()
        result = rag.answer(question)
        elapsed = time.time() - t

        print(f"  Retrieved passages  ({elapsed:.1f}s):")
        for rank, (p, s) in enumerate(
            zip(result.retrieved.passages, result.retrieved.scores), start=1
        ):
            snippet = p.replace("\n", " ")[:110]
            print(f"    [{rank}] {s:.3f}  {snippet}…")

        print(f"  Answer : {result.answer}")
        if gold:
            hit = any(g.lower() in result.answer.lower() for g in gold)
            tag = "✓" if hit else "✗"
            print(f"  Gold   : {gold}  {tag}")

    print(f"{'─' * 72}")
    print(f"\nTotal wall time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
