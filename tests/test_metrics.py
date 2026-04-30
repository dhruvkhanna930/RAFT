"""Tests for src/metrics/ — pure-Python components only.

Covers ASR, retrieval P/R/F1@k, stealth (visual diff + char entropy + PPL),
and efficiency metrics.  GPT-2 model calls are mocked — no downloads needed.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import pytest
import torch

from src.defenses.perplexity import PerplexityFilter
from src.metrics.asr import compute_asr, is_attack_successful
from src.metrics.efficiency import (
    EfficiencyRecord,
    compute_efficiency_metrics,
    timed_attack,
)
from src.metrics.retrieval import (
    compute_retrieval_metrics,
    mean_rank,
    precision_at_k,
    recall_at_k,
)
from src.metrics.stealth import char_class_entropy, compute_stealth_metrics, visual_diff_rate


# ── is_attack_successful ──────────────────────────────────────────────────────

class TestIsAttackSuccessful:
    def test_exact_match(self) -> None:
        assert is_attack_successful("Paris", "Paris")

    def test_substring_match(self) -> None:
        assert is_attack_successful("The capital is Paris.", "Paris")

    def test_case_insensitive_default(self) -> None:
        assert is_attack_successful("The answer is PARIS.", "paris")

    def test_case_sensitive_fails(self) -> None:
        assert not is_attack_successful("PARIS", "paris", case_sensitive=True)

    def test_case_sensitive_passes(self) -> None:
        assert is_attack_successful("Paris", "Paris", case_sensitive=True)

    def test_no_match(self) -> None:
        assert not is_attack_successful("London is the capital.", "Paris")

    def test_empty_target(self) -> None:
        # Empty string is always a substring
        assert is_attack_successful("anything", "")

    def test_empty_prediction(self) -> None:
        assert not is_attack_successful("", "Paris")


# ── compute_asr ───────────────────────────────────────────────────────────────

class TestComputeASR:
    def test_all_success(self) -> None:
        preds = ["Paris is the answer.", "Rome is the answer."]
        targets = ["paris", "rome"]
        assert compute_asr(preds, targets) == pytest.approx(1.0)

    def test_all_failure(self) -> None:
        preds = ["London", "Berlin"]
        targets = ["paris", "rome"]
        assert compute_asr(preds, targets) == pytest.approx(0.0)

    def test_half_success(self) -> None:
        preds = ["Paris", "Berlin"]
        targets = ["paris", "rome"]
        assert compute_asr(preds, targets) == pytest.approx(0.5)

    def test_empty_lists(self) -> None:
        assert compute_asr([], []) == pytest.approx(0.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_asr(["a", "b"], ["x"])


# ── precision_at_k / recall_at_k ─────────────────────────────────────────────

ADV = ["adv1", "adv2"]
RETRIEVED_FULL = ["adv1", "clean1", "adv2", "clean2", "clean3"]
RETRIEVED_MISS = ["clean1", "clean2", "clean3", "clean4", "clean5"]


class TestPrecisionRecall:
    def test_precision_perfect(self) -> None:
        # Both top-2 are adversarial
        assert precision_at_k(["adv1", "adv2", "clean1"], ADV, k=2) == pytest.approx(1.0)

    def test_precision_zero(self) -> None:
        assert precision_at_k(RETRIEVED_MISS, ADV, k=5) == pytest.approx(0.0)

    def test_precision_partial(self) -> None:
        # adv1 is at rank 1 out of k=5
        assert precision_at_k(RETRIEVED_FULL, ADV, k=5) == pytest.approx(2 / 5)

    def test_recall_perfect(self) -> None:
        # Both adversarial passages retrieved in top-2
        assert recall_at_k(["adv1", "adv2"], ADV, k=2) == pytest.approx(1.0)

    def test_recall_zero(self) -> None:
        assert recall_at_k(RETRIEVED_MISS, ADV, k=5) == pytest.approx(0.0)

    def test_recall_partial(self) -> None:
        # adv1 is retrieved but adv2 only at rank 3
        assert recall_at_k(["adv1", "clean1", "adv2"], ADV, k=2) == pytest.approx(0.5)

    def test_precision_k_zero(self) -> None:
        assert precision_at_k(RETRIEVED_FULL, ADV, k=0) == pytest.approx(0.0)

    def test_recall_empty_adversarial(self) -> None:
        assert recall_at_k(RETRIEVED_FULL, [], k=5) == pytest.approx(0.0)


# ── mean_rank ─────────────────────────────────────────────────────────────────

class TestMeanRank:
    def test_rank_1(self) -> None:
        assert mean_rank(["adv1", "clean1", "clean2"], ADV) == pytest.approx(1.0)

    def test_rank_3(self) -> None:
        assert mean_rank(["clean1", "clean2", "adv1"], ADV) == pytest.approx(3.0)

    def test_none_when_not_retrieved(self) -> None:
        assert mean_rank(["clean1", "clean2"], ADV) is None

    def test_first_adversarial_wins(self) -> None:
        # adv2 is at rank 2, adv1 at rank 4 — should return 2
        assert mean_rank(["clean1", "adv2", "clean2", "adv1"], ADV) == pytest.approx(2.0)


# ── compute_retrieval_metrics ─────────────────────────────────────────────────

class TestComputeRetrievalMetrics:
    def test_perfect_retrieval(self) -> None:
        retrieved = [["adv1", "adv2"]]
        adversarial = [["adv1", "adv2"]]
        m = compute_retrieval_metrics(retrieved, adversarial, k=2)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["mean_rank"] == pytest.approx(1.0)

    def test_no_retrieval(self) -> None:
        retrieved = [["c1", "c2"]]
        adversarial = [["adv1"]]
        m = compute_retrieval_metrics(retrieved, adversarial, k=2)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["mean_rank"] == float("inf")

    def test_empty_input(self) -> None:
        m = compute_retrieval_metrics([], [], k=5)
        assert m["precision"] == 0.0


# ── visual_diff_rate ──────────────────────────────────────────────────────────

class TestVisualDiffRate:
    def test_identical_strings(self) -> None:
        assert visual_diff_rate("hello world", "hello world") == pytest.approx(0.0)

    def test_zero_width_insertion_is_invisible(self) -> None:
        clean = "hello world"
        adv = "hello\u200b world"   # ZW space inserted — visually identical
        assert visual_diff_rate(clean, adv) == pytest.approx(0.0)

    def test_tag_char_insertion_is_invisible(self) -> None:
        tag = chr(0xE0061)  # tag 'a'
        clean = "hello world"
        adv = f"hel{tag}lo world"
        assert visual_diff_rate(clean, adv) == pytest.approx(0.0)

    def test_visible_change_nonzero(self) -> None:
        assert visual_diff_rate("hello", "world") > 0.0

    def test_partial_visible_change(self) -> None:
        rate = visual_diff_rate("hello world", "hello earth")
        assert 0.0 < rate <= 1.0


# ── char_class_entropy ────────────────────────────────────────────────────────

class TestCharClassEntropy:
    def test_pure_ascii(self) -> None:
        stats = char_class_entropy("hello world")
        assert stats["ascii_frac"] == pytest.approx(1.0)
        assert stats["zerowidth_frac"] == pytest.approx(0.0)

    def test_zero_width_fraction(self) -> None:
        # 1 ZW char in a 6-char string → zerowidth_frac = 1/6
        text = "hello\u200b"
        stats = char_class_entropy(text)
        assert stats["zerowidth_frac"] == pytest.approx(1 / 6)

    def test_tag_char_counted_as_zerowidth(self) -> None:
        tag = chr(0xE0061)
        text = f"hello{tag}"
        stats = char_class_entropy(text)
        assert stats["zerowidth_frac"] == pytest.approx(1 / 6)

    def test_fractions_sum_to_one(self) -> None:
        stats = char_class_entropy("hello\u200b world!")
        total = sum(stats.values())
        assert total == pytest.approx(1.0)

    def test_empty_string(self) -> None:
        stats = char_class_entropy("")
        assert all(v == 0.0 for v in stats.values())


# ── EfficiencyRecord + compute_efficiency_metrics ────────────────────────────

class TestEfficiencyMetrics:
    def test_defaults(self) -> None:
        rec = EfficiencyRecord()
        assert rec.llm_queries == 0
        assert rec.runtime_seconds == 0.0

    def test_compute_empty(self) -> None:
        m = compute_efficiency_metrics([])
        assert m["avg_llm_queries"] == 0.0
        assert m["avg_runtime_s"] == 0.0

    def test_compute_averages(self) -> None:
        records = [
            EfficiencyRecord(llm_queries=10, runtime_seconds=1.0, perturbation_count=5, iterations=20),
            EfficiencyRecord(llm_queries=20, runtime_seconds=3.0, perturbation_count=15, iterations=40),
        ]
        m = compute_efficiency_metrics(records)
        assert m["avg_llm_queries"] == pytest.approx(15.0)
        assert m["avg_runtime_s"] == pytest.approx(2.0)
        assert m["avg_perturbation_count"] == pytest.approx(10.0)
        assert m["avg_iterations"] == pytest.approx(30.0)

    def test_timed_attack_records_time(self) -> None:
        with timed_attack() as rec:
            time.sleep(0.05)
        assert rec.runtime_seconds >= 0.04  # at least 40ms

    def test_timed_attack_fields_writable(self) -> None:
        with timed_attack() as rec:
            rec.llm_queries = 7
            rec.perturbation_count = 12
        assert rec.llm_queries == 7
        assert rec.perturbation_count == 12


# ── PerplexityFilter (GPT-2 scorer, mocked) ───────────────────────────────────

def _make_ppl_scorer(loss_val: float = 2.0, threshold: float = 50.0) -> PerplexityFilter:
    """Build a PerplexityFilter with a mock model that returns *loss_val*."""
    fake_out = MagicMock()
    fake_out.loss = torch.tensor(loss_val)

    fake_model = MagicMock()
    fake_model.config.max_position_embeddings = 1024
    fake_model.return_value = fake_out  # model(input_ids, labels=…) → fake_out

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": torch.ones(1, 5, dtype=torch.long)}

    scorer = PerplexityFilter(threshold=threshold)
    scorer._model = fake_model       # bypass lazy _load_model
    scorer._tokenizer = fake_tokenizer
    return scorer


class TestPerplexityScorer:
    def test_lazy_load_not_called_at_init(self) -> None:
        scorer = PerplexityFilter()
        assert scorer._model is None

    def test_score_exp_of_loss(self) -> None:
        # loss = 2.0  →  PPL = exp(2.0) ≈ 7.389
        scorer = _make_ppl_scorer(loss_val=2.0)
        assert scorer.score("hello world") == pytest.approx(math.exp(2.0), rel=1e-4)

    def test_score_loss_zero_gives_ppl_one(self) -> None:
        scorer = _make_ppl_scorer(loss_val=0.0)
        assert scorer.score("hello") == pytest.approx(1.0, rel=1e-4)

    def test_score_empty_passage_returns_zero(self) -> None:
        scorer = PerplexityFilter()
        fake_tokenizer = MagicMock()
        fake_tokenizer.return_value = {"input_ids": torch.zeros(1, 0, dtype=torch.long)}
        fake_model = MagicMock()
        fake_model.config.max_position_embeddings = 1024
        scorer._model = fake_model
        scorer._tokenizer = fake_tokenizer
        assert scorer.score("") == 0.0

    def test_apply_keeps_passage_below_threshold(self) -> None:
        # exp(2.0) ≈ 7.4 < 50.0 → passage kept
        scorer = _make_ppl_scorer(loss_val=2.0, threshold=50.0)
        assert scorer.apply("some passage") == "some passage"

    def test_apply_rejects_passage_above_threshold(self) -> None:
        # exp(4.0) ≈ 54.6 > 50.0 → passage rejected → ""
        scorer = _make_ppl_scorer(loss_val=4.0, threshold=50.0)
        assert scorer.apply("bad passage") == ""

    def test_apply_list_filters_high_ppl(self) -> None:
        # All passages share the same mock loss, so all pass or all fail together.
        scorer = _make_ppl_scorer(loss_val=2.0, threshold=50.0)
        result = scorer.apply(["pass1", "pass2", "pass3"])
        assert result == ["pass1", "pass2", "pass3"]

    def test_apply_list_empty(self) -> None:
        scorer = _make_ppl_scorer(loss_val=2.0)
        assert scorer.apply([]) == []

    def test_apply_list_all_rejected(self) -> None:
        scorer = _make_ppl_scorer(loss_val=5.0, threshold=5.0)
        # exp(5.0) ≈ 148 > 5.0
        result = scorer.apply(["a", "b"])
        assert result == []


# ── compute_stealth_metrics ───────────────────────────────────────────────────

class TestComputeStealthMetrics:
    _CLEAN = ["hello world", "the quick brown fox"]
    _ADV = ["hello\u200b world", "the quick brown fox"]  # one ZW insertion, one identical

    def test_basic_keys_without_ppl(self) -> None:
        m = compute_stealth_metrics(self._ADV, self._CLEAN)
        assert "visual_diff_rate" in m
        assert "avg_zerowidth_frac" in m
        assert "avg_nonascii_frac" in m
        assert "avg_ppl_adv" not in m

    def test_ppl_keys_present_with_scorer(self) -> None:
        scorer = _make_ppl_scorer(loss_val=2.0)
        m = compute_stealth_metrics(self._ADV, self._CLEAN, ppl_scorer=scorer)
        assert "avg_ppl_adv" in m
        assert "avg_ppl_clean" in m
        assert "ppl_ratio" in m

    def test_ppl_values_correct(self) -> None:
        scorer = _make_ppl_scorer(loss_val=2.0)
        m = compute_stealth_metrics(self._ADV, self._CLEAN, ppl_scorer=scorer)
        expected = math.exp(2.0)
        assert m["avg_ppl_adv"] == pytest.approx(expected, rel=1e-4)
        assert m["avg_ppl_clean"] == pytest.approx(expected, rel=1e-4)
        assert m["ppl_ratio"] == pytest.approx(1.0, rel=1e-4)

    def test_invisible_insertion_zero_visual_diff(self) -> None:
        # ZW space is invisible → visual_diff_rate should be 0
        m = compute_stealth_metrics(
            ["hello\u200b world"],
            ["hello world"],
        )
        assert m["visual_diff_rate"] == pytest.approx(0.0)

    def test_nonzero_zerowidth_frac(self) -> None:
        m = compute_stealth_metrics(
            ["hello\u200b"],   # 1 ZW in 6 chars → frac = 1/6
            ["hello!"],
        )
        assert m["avg_zerowidth_frac"] == pytest.approx(1 / 6, rel=1e-4)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_stealth_metrics(["a", "b"], ["x"])
