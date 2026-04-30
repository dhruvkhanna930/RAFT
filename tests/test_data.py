"""Tests for the data loading and indexing layer.

Three test classes:
1. TestNQLoader        — JSONL I/O and corpus_size / n_questions limits
2. TestIndexBuilder    — FAISS build, cache, load, search (stub retriever, no ML)
3. TestContrieverRetriever — encode + retrieve logic (fake model, no downloads)

All tests run offline with no network or GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.data.index_builder import IndexBuilder, _load_corpus_jsonl, _save_corpus_jsonl
from src.data.nq_loader import NQLoader
from src.retrievers.contriever import ContrieverRetriever, _autodetect_device

# ── Shared helpers ────────────────────────────────────────────────────────────

_DIM = 64  # small dimension for fast tests


def _make_passages(n: int) -> list[str]:
    return [f"This is passage number {i} about various topics." for i in range(n)]


# ── Stub retriever for IndexBuilder tests (no ML) ────────────────────────────

class _StubRetriever:
    """Returns deterministic random embeddings — no model loading."""

    def encode_passages(self, passages: list[str], batch_size: int = 64) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(passages), _DIM)).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.standard_normal(_DIM).astype(np.float32)


# ── Fake model helpers for ContrieverRetriever tests ─────────────────────────

class _FakeModelOutput:
    def __init__(self, batch_size: int, dim: int = _DIM, seq_len: int = 8) -> None:
        self.last_hidden_state = torch.randn(batch_size, seq_len, dim)


class _FakeModel:
    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def __call__(self, input_ids: Any = None, attention_mask: Any = None, **_: Any) -> _FakeModelOutput:
        n = input_ids.shape[0]
        return _FakeModelOutput(n, self.dim)

    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self


class _FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        n = len(texts)
        return {
            "input_ids": torch.ones(n, 8, dtype=torch.long),
            "attention_mask": torch.ones(n, 8, dtype=torch.long),
        }


def _make_retriever_with_fake_model(dim: int = _DIM) -> ContrieverRetriever:
    """Return a ContrieverRetriever with a fake model injected (no downloads)."""
    r = ContrieverRetriever(device="cpu", normalize=True)
    r._model = _FakeModel(dim)
    r._tokenizer = _FakeTokenizer()
    return r


# ════════════════════════════════════════════════════════════════════════════
# 1. NQLoader
# ════════════════════════════════════════════════════════════════════════════


def _fake_corpus_dataset(n: int = 12) -> list[dict[str, Any]]:
    return [{"_id": str(i), "title": f"Title {i}", "text": f"Passage text {i}."} for i in range(n)]


def _fake_questions_dataset(n: int = 10) -> list[dict[str, str | list[str]]]:
    return [{"question": f"Question {i}?", "answer": [f"Answer {i}"]} for i in range(n)]


class TestNQLoader:
    def test_corpus_jsonl_written(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path, corpus_size=-1)
        with patch("src.data.nq_loader._load_hf_dataset", return_value=_fake_corpus_dataset(5)):
            loader._download_corpus()
        assert loader.corpus_path.exists()
        lines = loader.corpus_path.read_text().strip().splitlines()
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert "id" in first and "text" in first

    def test_questions_jsonl_written(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path)
        with patch("src.data.nq_loader._load_hf_dataset", return_value=_fake_questions_dataset(7)):
            loader._download_questions()
        lines = loader.questions_path.read_text().strip().splitlines()
        assert len(lines) == 7
        first = json.loads(lines[0])
        assert "question" in first and "answers" in first

    def test_corpus_size_limit_respected(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path, corpus_size=3)
        with patch("src.data.nq_loader._load_hf_dataset", return_value=_fake_corpus_dataset(10)):
            loader._download_corpus()
        lines = loader.corpus_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_n_questions_limit_respected(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path, n_questions=4)
        with patch("src.data.nq_loader._load_hf_dataset", return_value=_fake_questions_dataset(10)):
            loader._download_questions()
        lines = loader.questions_path.read_text().strip().splitlines()
        assert len(lines) == 4

    def test_passages_returns_text_strings(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path, corpus_size=-1)
        with patch("src.data.nq_loader._load_hf_dataset", return_value=_fake_corpus_dataset(5)):
            ps = loader.passages()
        assert isinstance(ps, list)
        assert all(isinstance(p, str) for p in ps)
        assert len(ps) == 5

    def test_cached_corpus_not_redownloaded(self, tmp_path: Path) -> None:
        """If both cache files already exist, _load_hf_dataset must not be called."""
        loader = NQLoader(processed_dir=tmp_path, corpus_size=3, n_questions=2)
        loader._nq_dir.mkdir(parents=True)
        loader.corpus_path.write_text(
            "\n".join(json.dumps({"id": str(i), "title": "", "text": f"p{i}"}) for i in range(3))
            + "\n"
        )
        loader.questions_path.write_text(
            "\n".join(
                json.dumps({"id": str(i), "question": f"q{i}?", "answers": [f"a{i}"]})
                for i in range(2)
            )
            + "\n"
        )
        mock_ds = MagicMock()
        with patch("src.data.nq_loader._load_hf_dataset", mock_ds):
            corpus, questions = loader.load()
        mock_ds.assert_not_called()
        assert len(corpus) == 3
        assert len(questions) == 2

    def test_read_jsonl_limit(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path)
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps({"text": f"p{i}"}) for i in range(10)) + "\n")
        result = list(loader._read_jsonl(path, limit=4))
        assert len(result) == 4

    def test_read_jsonl_no_limit(self, tmp_path: Path) -> None:
        loader = NQLoader(processed_dir=tmp_path)
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps({"text": f"p{i}"}) for i in range(6)) + "\n")
        result = list(loader._read_jsonl(path, limit=-1))
        assert len(result) == 6


# ════════════════════════════════════════════════════════════════════════════
# 2. IndexBuilder
# ════════════════════════════════════════════════════════════════════════════


class TestIndexBuilder:
    def test_build_creates_faiss_index(self, tmp_path: Path) -> None:
        passages = _make_passages(100)
        builder = IndexBuilder(indices_dir=tmp_path, dataset="nq", retriever_name="stub")
        index, corpus = builder.build(passages, _StubRetriever())
        assert index.ntotal == 100
        assert len(corpus) == 100

    def test_build_writes_cache_files(self, tmp_path: Path) -> None:
        passages = _make_passages(50)
        builder = IndexBuilder(indices_dir=tmp_path)
        builder.build(passages, _StubRetriever())
        assert builder.index_path(50).exists()
        assert builder.corpus_path(50).exists()

    def test_build_loads_from_cache_second_call(self, tmp_path: Path) -> None:
        """Second build() call must not call encode_passages again."""
        passages = _make_passages(30)
        call_count = [0]

        class _CountingRetriever(_StubRetriever):
            def encode_passages(self, ps: list[str], batch_size: int = 64) -> np.ndarray:
                call_count[0] += 1
                return super().encode_passages(ps, batch_size)

        builder = IndexBuilder(indices_dir=tmp_path)
        builder.build(passages, _CountingRetriever())
        builder.build(passages, _CountingRetriever())  # should use cache
        assert call_count[0] == 1

    def test_force_rebuild_bypasses_cache(self, tmp_path: Path) -> None:
        passages = _make_passages(20)
        call_count = [0]

        class _CountingRetriever(_StubRetriever):
            def encode_passages(self, ps: list[str], batch_size: int = 64) -> np.ndarray:
                call_count[0] += 1
                return super().encode_passages(ps, batch_size)

        builder = IndexBuilder(indices_dir=tmp_path)
        builder.build(passages, _CountingRetriever())
        builder.build(passages, _CountingRetriever(), force=True)
        assert call_count[0] == 2

    def test_load_raises_if_no_cache(self, tmp_path: Path) -> None:
        builder = IndexBuilder(indices_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            builder.load(corpus_size=999)

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        passages = _make_passages(40)
        builder = IndexBuilder(indices_dir=tmp_path)
        builder.build(passages, _StubRetriever())
        index, loaded = builder.load(corpus_size=40)
        assert index.ntotal == 40
        assert loaded == passages

    def test_search_returns_k_results(self, tmp_path: Path) -> None:
        passages = _make_passages(100)
        retriever = _StubRetriever()
        builder = IndexBuilder(indices_dir=tmp_path)
        index, corpus = builder.build(passages, retriever)
        query_vec = retriever.encode_query("some question")
        results, scores = builder.search(query_vec, index, corpus, k=5)
        assert len(results) == 5
        assert len(scores) == 5

    def test_search_scores_sorted_descending(self, tmp_path: Path) -> None:
        passages = _make_passages(50)
        retriever = _StubRetriever()
        builder = IndexBuilder(indices_dir=tmp_path)
        index, corpus = builder.build(passages, retriever)
        query_vec = retriever.encode_query("question")
        _, scores = builder.search(query_vec, index, corpus, k=5)
        assert scores == sorted(scores, reverse=True)

    def test_cache_key_naming(self) -> None:
        builder = IndexBuilder(dataset="nq", retriever_name="contriever")
        assert builder.cache_key(1000) == "nq_contriever_1000"
        assert builder.cache_key(-1) == "nq_contriever_full"

    def test_corpus_jsonl_roundtrip(self, tmp_path: Path) -> None:
        passages = ["hello world", "foo bar baz", "unicode: café"]
        path = tmp_path / "corpus.jsonl"
        _save_corpus_jsonl(passages, path)
        loaded = _load_corpus_jsonl(path)
        assert loaded == passages


# ════════════════════════════════════════════════════════════════════════════
# 3. ContrieverRetriever
# ════════════════════════════════════════════════════════════════════════════


class TestContrieverRetriever:
    def test_autodetect_device_returns_string(self) -> None:
        device = _autodetect_device()
        assert device in {"cpu", "cuda", "mps"}

    def test_auto_device_set_on_init(self) -> None:
        r = ContrieverRetriever(device="auto")
        assert r.device in {"cpu", "cuda", "mps"}

    def test_explicit_device_preserved(self) -> None:
        r = ContrieverRetriever(device="cpu")
        assert r.device == "cpu"

    def test_encode_query_shape(self) -> None:
        r = _make_retriever_with_fake_model()
        vec = r.encode_query("What is the capital of France?")
        assert vec.ndim == 1
        assert vec.shape[0] == _DIM

    def test_encode_query_is_float32(self) -> None:
        r = _make_retriever_with_fake_model()
        vec = r.encode_query("test")
        assert vec.dtype == np.float32

    def test_encode_passages_shape(self) -> None:
        r = _make_retriever_with_fake_model()
        vecs = r.encode_passages(_make_passages(10))
        assert vecs.shape == (10, _DIM)

    def test_encode_passages_normalized_l2(self) -> None:
        r = _make_retriever_with_fake_model()
        vecs = r.encode_passages(_make_passages(5))
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)

    def test_embed_alias(self) -> None:
        r = _make_retriever_with_fake_model()
        a = r.embed(_make_passages(3))
        b = r.encode_passages(_make_passages(3))
        # same shape — values differ due to randomness in fake model, just check shape
        assert a.shape == b.shape

    def test_build_index_sets_corpus(self) -> None:
        r = _make_retriever_with_fake_model()
        passages = _make_passages(20)
        r.build_index(passages)
        assert r._corpus == passages

    def test_build_index_creates_faiss_index(self) -> None:
        r = _make_retriever_with_fake_model()
        passages = _make_passages(20)
        r.build_index(passages)
        assert r._index is not None
        assert r._index.ntotal == 20

    def test_retrieve_raises_before_build_index(self) -> None:
        r = _make_retriever_with_fake_model()
        with pytest.raises(RuntimeError):
            r.retrieve("some query")

    def test_retrieve_returns_k_results(self) -> None:
        r = _make_retriever_with_fake_model()
        passages = _make_passages(30)
        r.build_index(passages)
        results, scores = r.retrieve("question about passage", k=5)
        assert len(results) == 5
        assert len(scores) == 5

    def test_retrieve_results_are_passage_strings(self) -> None:
        r = _make_retriever_with_fake_model()
        passages = _make_passages(10)
        r.build_index(passages)
        results, _ = r.retrieve("query", k=3)
        for p in results:
            assert p in passages

    def test_mean_pool_shape(self) -> None:
        batch, seq_len, dim = 4, 8, 32
        hidden = torch.randn(batch, seq_len, dim)
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        pooled = ContrieverRetriever._mean_pool(hidden, mask)
        assert pooled.shape == (batch, dim)

    def test_mean_pool_masks_out_padding(self) -> None:
        """Pooling should ignore padded (mask=0) positions."""
        batch, seq_len, dim = 1, 4, 8
        hidden = torch.zeros(batch, seq_len, dim)
        hidden[0, 0, :] = 1.0   # only first token has signal
        mask = torch.tensor([[1, 0, 0, 0]])  # pad all but first token
        pooled = ContrieverRetriever._mean_pool(hidden, mask)
        np.testing.assert_allclose(pooled.numpy(), np.ones((1, dim)), atol=1e-5)
