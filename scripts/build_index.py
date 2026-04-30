#!/usr/bin/env python3
"""CLI script: build and cache a FAISS index for a dataset corpus.

Usage examples::

    # Build a 1 000-passage dev index for NQ with Contriever
    python scripts/build_index.py --dataset nq --corpus-size 1000 --retriever contriever

    # Full corpus (warning: 2.6 M passages, takes ~1 h on CPU)
    python scripts/build_index.py --dataset nq --retriever contriever

    # Force rebuild even if a cache already exists
    python scripts/build_index.py --dataset nq --corpus-size 1000 --force

After the index is built the script prints:
- Number of passages indexed
- Cache file location
- Top-5 results for a sample query
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.index_builder import IndexBuilder
from src.data.nq_loader import NQLoader
from src.retrievers.contriever import ContrieverRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")

_SAMPLE_QUERIES = {
    "nq": "What is the capital of France?",
    "hotpotqa": "Who was the first president of the United States?",
    "msmarco": "What is the speed of light?",
}

_RETRIEVER_MAP = {
    "contriever": ContrieverRetriever,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a FAISS index for a dataset corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="nq",
        choices=["nq", "hotpotqa", "msmarco"],
        help="Dataset to index",
    )
    p.add_argument(
        "--corpus-size",
        type=int,
        default=-1,
        metavar="N",
        help="Cap corpus at N passages (-1 = full)",
    )
    p.add_argument(
        "--retriever",
        default="contriever",
        choices=list(_RETRIEVER_MAP),
        help="Retriever / encoder to use",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Encoding batch size",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if a cached index already exists",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (contains processed/ and indices/)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)

    # ── Load passages ─────────────────────────────────────────────────────────
    if args.dataset == "nq":
        loader = NQLoader(
            processed_dir=data_dir / "processed",
            corpus_size=args.corpus_size,
        )
        passages = loader.passages()
    else:
        logger.error("Loader for dataset '%s' not yet implemented.", args.dataset)
        sys.exit(1)

    logger.info("Passages loaded: %d", len(passages))

    # ── Build retriever ───────────────────────────────────────────────────────
    retriever_cls = _RETRIEVER_MAP[args.retriever]
    retriever = retriever_cls(device="auto", batch_size=args.batch_size)
    logger.info("Retriever: %s  device=%s", retriever_cls.__name__, retriever.device)

    # ── Build index ───────────────────────────────────────────────────────────
    builder = IndexBuilder(
        indices_dir=data_dir / "indices",
        dataset=args.dataset,
        retriever_name=args.retriever,
    )
    index, corpus = builder.build(
        passages, retriever, batch_size=args.batch_size, force=args.force
    )
    logger.info(
        "Index ready: %d vectors  →  %s",
        index.ntotal,
        builder.index_path(len(corpus)),
    )

    # ── Sample retrieval ──────────────────────────────────────────────────────
    sample_query = _SAMPLE_QUERIES.get(args.dataset, "What is Wikipedia?")
    query_vec = retriever.encode_query(sample_query)
    results, scores = builder.search(query_vec, index, corpus, k=5)

    print(f"\nSample query: {sample_query!r}")
    print("-" * 72)
    for rank, (passage, score) in enumerate(zip(results, scores), start=1):
        snippet = passage.replace("\n", " ")[:100]
        print(f"  [{rank}] score={score:.4f}  {snippet}…")


if __name__ == "__main__":
    main()
