"""NQ corpus and question loader (HuggingFace / BEIR format).

Downloads the NQ passage corpus from ``BeIR/nq`` and questions from
``nq_open``, then caches them as JSONL files under *processed_dir*/nq/.

Scale is controlled by the two sentinel fields from :class:`DatasetCfg`:
- ``corpus_size = -1``  → load the full corpus
- ``corpus_size = N``   → cap at the first N passages (dev / smoke-test)
- ``n_questions = -1``  → load all questions
- ``n_questions = N``   → cap at N questions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

logger = logging.getLogger(__name__)

_CORPUS_HF_ID = "BeIR/nq"
_QUESTIONS_HF_ID = "nq_open"


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    """Thin wrapper around ``datasets.load_dataset`` so tests can mock it.

    Using a module-level function (rather than an inline lazy import) lets
    :func:`unittest.mock.patch` target ``src.data.nq_loader._load_hf_dataset``
    without requiring ``datasets`` to be importable at patch time.
    """
    from datasets import load_dataset  # lazy — keep 'datasets' as optional dep

    return load_dataset(*args, **kwargs)


class NQLoader:
    """Load and cache the NQ dataset (BEIR corpus + NQ Open questions).

    Files are written once to ``processed_dir/nq/corpus.jsonl`` and
    ``processed_dir/nq/questions.jsonl``.  Subsequent calls read from
    cache unless ``force=True``.

    Args:
        processed_dir: Root directory for processed data (e.g. ``data/processed``).
        corpus_size: Max passages to load; ``-1`` means all.
        n_questions: Max questions to load; ``-1`` means all.
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        corpus_size: int = -1,
        n_questions: int = -1,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.corpus_size = corpus_size
        self.n_questions = n_questions
        self._nq_dir = self.processed_dir / "nq"

    # ── Public paths ──────────────────────────────────────────────────────────

    @property
    def corpus_path(self) -> Path:
        """Path to the cached corpus JSONL file."""
        return self._nq_dir / "corpus.jsonl"

    @property
    def questions_path(self) -> Path:
        """Path to the cached questions JSONL file."""
        return self._nq_dir / "questions.jsonl"

    # ── Public methods ────────────────────────────────────────────────────────

    def load(
        self, force: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load corpus and questions, downloading if needed.

        Args:
            force: Re-download even if cache files already exist.

        Returns:
            Tuple of (corpus_records, question_records).  Each record is
            a plain dict with at minimum a ``"text"`` key (corpus) or
            ``"question"`` / ``"answers"`` keys (questions).
        """
        if force or not self.corpus_path.exists():
            self._download_corpus()
        if force or not self.questions_path.exists():
            self._download_questions()

        corpus = list(self._read_jsonl(self.corpus_path, limit=self.corpus_size))
        questions = list(self._read_jsonl(self.questions_path, limit=self.n_questions))
        logger.info("Loaded %d passages, %d questions", len(corpus), len(questions))
        return corpus, questions

    def passages(self, force: bool = False) -> list[str]:
        """Return passage text strings suitable for passing to a retriever.

        Args:
            force: Re-download if cache is missing.

        Returns:
            List of passage text strings.
        """
        corpus, _ = self.load(force=force)
        return [r["text"] for r in corpus]

    def questions(self, force: bool = False) -> list[dict[str, Any]]:
        """Return question records.

        Args:
            force: Re-download if cache is missing.

        Returns:
            List of question dicts (keys: ``id``, ``question``, ``answers``).
        """
        _, qs = self.load(force=force)
        return qs

    # ── Private ───────────────────────────────────────────────────────────────

    def _download_corpus(self) -> None:
        """Download the BEIR NQ corpus from HuggingFace and write to JSONL."""
        self._nq_dir.mkdir(parents=True, exist_ok=True)
        limit = None if self.corpus_size == -1 else self.corpus_size
        logger.info("Downloading NQ corpus (limit=%s)…", limit or "full")
        dataset = _load_hf_dataset(_CORPUS_HF_ID, "corpus", split="corpus")
        with self.corpus_path.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(tqdm(dataset, desc="NQ corpus")):
                if limit is not None and i >= limit:
                    break
                record: dict[str, Any] = {
                    "id": row.get("_id", str(i)),
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Corpus written to %s", self.corpus_path)

    def _download_questions(self) -> None:
        """Download NQ Open questions from HuggingFace and write to JSONL."""
        self._nq_dir.mkdir(parents=True, exist_ok=True)
        limit = None if self.n_questions == -1 else self.n_questions
        logger.info("Downloading NQ questions (limit=%s)…", limit or "all")
        dataset = _load_hf_dataset(_QUESTIONS_HF_ID, split="train")
        with self.questions_path.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(tqdm(dataset, desc="NQ questions")):
                if limit is not None and i >= limit:
                    break
                record = {
                    "id": str(i),
                    "question": row.get("question", ""),
                    "answers": row.get("answer", []),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Questions written to %s", self.questions_path)

    @staticmethod
    def _read_jsonl(path: Path, limit: int = -1) -> Iterator[dict[str, Any]]:
        """Iterate over records in a JSONL file, optionally capping at *limit*.

        Args:
            path: Path to a ``.jsonl`` file.
            limit: Maximum records to yield; ``-1`` means unlimited.

        Yields:
            Parsed JSON dicts.
        """
        count = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if limit != -1 and count >= limit:
                    break
                yield json.loads(line)
                count += 1
