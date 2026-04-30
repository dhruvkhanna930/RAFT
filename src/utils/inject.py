"""Corpus injection utility.

Appends adversarial passages to an existing JSONL corpus file and writes
the poisoned corpus to a new path.  Optionally triggers a retriever index
rebuild so the poisoned corpus is immediately searchable.

Intended use::

    poisoned = inject_passages(
        corpus_path   = "data/processed/nq/corpus.jsonl",
        malicious_passages = adv_passages,
        output_path   = "data/poisoned/nq/poisoned_corpus.jsonl",
        retriever     = retriever,          # triggers build_index
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def inject_passages(
    corpus_path: str | Path,
    malicious_passages: list[str],
    output_path: str | Path,
    retriever: Any | None = None,
) -> list[str]:
    """Append *malicious_passages* to the corpus at *corpus_path* and write output.

    Args:
        corpus_path: Path to the clean JSONL corpus
            (one ``{"text": "..."}`` record per line).
        malicious_passages: Adversarial passage strings to inject.
        output_path: Destination path for the poisoned JSONL corpus.
        retriever: Optional retriever instance.  If provided,
            ``retriever.build_index(poisoned_passages)`` is called after
            writing so the index is ready for immediate querying.

    Returns:
        Full poisoned passage list (original passages + malicious passages).
    """
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)

    # ── Read clean corpus ─────────────────────────────────────────────────────
    clean_passages: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            clean_passages.append(record["text"])

    poisoned = clean_passages + malicious_passages
    logger.info(
        "inject_passages: %d clean + %d adversarial = %d total",
        len(clean_passages),
        len(malicious_passages),
        len(poisoned),
    )

    # ── Write poisoned corpus ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, text in enumerate(clean_passages):
            fh.write(
                json.dumps({"id": str(i), "text": text}, ensure_ascii=False) + "\n"
            )
        for j, text in enumerate(malicious_passages):
            fh.write(
                json.dumps(
                    {"id": f"adv_{j}", "text": text, "is_adversarial": True},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Poisoned corpus written to %s", output_path)

    # ── Optionally rebuild index ───────────────────────────────────────────────
    if retriever is not None:
        logger.info("Rebuilding index over %d passages…", len(poisoned))
        retriever.build_index(poisoned)

    return poisoned
