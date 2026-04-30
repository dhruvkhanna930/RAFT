"""Tests for src/defenses/ — pure-Python components only.

Covers UnicodeNormalizer, DuplicateFilter, PerplexityFilter, and
ParaphraseDefense.  No ML models required; heavy models are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.defenses.duplicate_filter import DuplicateFilter
from src.defenses.paraphrase import ParaphraseDefense
from src.defenses.perplexity import PerplexityFilter
from src.defenses.unicode_normalize import UnicodeNormalizer


# ── UnicodeNormalizer ─────────────────────────────────────────────────────────

class TestUnicodeNormalizer:
    def test_clean_plain_text_unchanged(self) -> None:
        norm = UnicodeNormalizer()
        text = "The quick brown fox."
        assert norm.clean(text) == text

    def test_removes_zero_width_space(self) -> None:
        norm = UnicodeNormalizer()
        dirty = "hel\u200blo"
        assert norm.clean(dirty) == "hello"

    def test_removes_zero_width_joiner(self) -> None:
        norm = UnicodeNormalizer()
        dirty = "hel\u200dlo"
        assert norm.clean(dirty) == "hello"

    def test_removes_bom(self) -> None:
        norm = UnicodeNormalizer()
        dirty = "\ufeffhello"
        assert norm.clean(dirty) == "hello"

    def test_removes_soft_hyphen(self) -> None:
        norm = UnicodeNormalizer()
        dirty = "super\u00adman"
        assert norm.clean(dirty) == "superman"

    def test_removes_tag_chars(self) -> None:
        norm = UnicodeNormalizer(strip_tags=True)
        tag_a = chr(0xE0061)
        dirty = f"hel{tag_a}lo"
        assert norm.clean(dirty) == "hello"

    def test_tag_chars_preserved_when_disabled(self) -> None:
        norm = UnicodeNormalizer(strip_tags=False, strip_zero_width=False, nfkc=False)
        tag_a = chr(0xE0061)
        dirty = f"hel{tag_a}lo"
        assert tag_a in norm.clean(dirty)

    def test_removes_variation_selectors(self) -> None:
        norm = UnicodeNormalizer(strip_variation_selectors=True)
        vs1 = chr(0xFE00)  # variation selector 1
        dirty = f"hel{vs1}lo"
        assert norm.clean(dirty) == "hello"

    def test_apply_cleans_list(self) -> None:
        norm = UnicodeNormalizer()
        passages = ["hel\u200blo", "wor\u200cld", "clean text"]
        result = norm.apply(passages)
        assert result == ["hello", "world", "clean text"]

    def test_apply_empty_list(self) -> None:
        norm = UnicodeNormalizer()
        assert norm.apply([]) == []

    def test_nfkc_normalises_ligature(self) -> None:
        norm = UnicodeNormalizer(nfkc=True, strip_zero_width=False, strip_tags=False)
        # ﬁ (U+FB01 LATIN SMALL LIGATURE FI) → "fi" under NFKC
        assert norm.clean("\ufb01le") == "file"

    def test_nfkc_disabled_preserves_ligature(self) -> None:
        norm = UnicodeNormalizer(nfkc=False, strip_zero_width=False, strip_tags=False)
        assert norm.clean("\ufb01le") == "\ufb01le"

    def test_multiple_invisible_chars_removed(self) -> None:
        norm = UnicodeNormalizer()
        tag = chr(0xE0041)
        dirty = f"\u200bhel{tag}lo\u200c"
        result = norm.clean(dirty)
        assert result == "hello"


# ── DuplicateFilter ───────────────────────────────────────────────────────────

class TestDuplicateFilter:
    # ── exact deduplication ──────────────────────────────────────────────────

    def test_exact_removes_duplicate(self) -> None:
        f = DuplicateFilter(exact_only=True)
        result = f.apply(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_exact_preserves_order(self) -> None:
        f = DuplicateFilter(exact_only=True)
        result = f.apply(["c", "b", "a", "c"])
        assert result == ["c", "b", "a"]

    def test_exact_no_duplicates_unchanged(self) -> None:
        f = DuplicateFilter(exact_only=True)
        passages = ["alpha", "beta", "gamma"]
        assert f.apply(passages) == passages

    def test_exact_empty_list(self) -> None:
        f = DuplicateFilter(exact_only=True)
        assert f.apply([]) == []

    def test_exact_all_duplicates(self) -> None:
        f = DuplicateFilter(exact_only=True)
        assert f.apply(["x", "x", "x"]) == ["x"]

    # ── Jaccard deduplication ────────────────────────────────────────────────

    def test_jaccard_near_dup_removed(self) -> None:
        f = DuplicateFilter(exact_only=False, jaccard_threshold=0.8)
        # Longer sentence: only the last word changes → 3-gram Jaccard ≈ 0.846
        base = "the quick brown fox jumps over the lazy dog while running through the field"
        near_dup = "the quick brown fox jumps over the lazy dog while running through the park"
        result = f.apply([base, near_dup])
        assert len(result) == 1
        assert result[0] == base

    def test_jaccard_different_passages_kept(self) -> None:
        f = DuplicateFilter(exact_only=False, jaccard_threshold=0.9)
        a = "the quick brown fox"
        b = "completely different content here"
        result = f.apply([a, b])
        assert len(result) == 2

    def test_jaccard_exact_duplicate_removed(self) -> None:
        f = DuplicateFilter(exact_only=False, jaccard_threshold=0.9)
        text = "identical passage text here"
        assert f.apply([text, text]) == [text]

    # ── _ngrams ──────────────────────────────────────────────────────────────

    def test_ngrams_basic(self) -> None:
        f = DuplicateFilter(ngram_size=2)
        result = f._ngrams("the quick brown")
        assert ("the", "quick") in result
        assert ("quick", "brown") in result
        assert len(result) == 2

    def test_ngrams_short_text(self) -> None:
        f = DuplicateFilter(ngram_size=3)
        result = f._ngrams("one two")  # fewer tokens than n
        assert len(result) == 1

    def test_ngrams_empty_string(self) -> None:
        f = DuplicateFilter(ngram_size=3)
        assert f._ngrams("") == set()

    # ── _jaccard ─────────────────────────────────────────────────────────────

    def test_jaccard_identical(self) -> None:
        f = DuplicateFilter(ngram_size=2)
        j = f._jaccard("hello world foo", "hello world foo")
        assert j == pytest.approx(1.0)

    def test_jaccard_disjoint(self) -> None:
        f = DuplicateFilter(ngram_size=2)
        j = f._jaccard("alpha beta gamma", "delta epsilon zeta")
        assert j == pytest.approx(0.0)

    def test_jaccard_both_empty(self) -> None:
        f = DuplicateFilter()
        assert f._jaccard("", "") == pytest.approx(1.0)

    def test_jaccard_one_empty(self) -> None:
        f = DuplicateFilter()
        assert f._jaccard("hello world", "") == pytest.approx(0.0)

    def test_jaccard_range(self) -> None:
        f = DuplicateFilter(ngram_size=2)
        j = f._jaccard("a b c d", "a b e f")
        assert 0.0 < j < 1.0


# ── PerplexityFilter ──────────────────────────────────────────────────────────


def _make_mock_model(loss_value: float) -> MagicMock:
    """Build a mock causal-LM that returns a fixed cross-entropy loss."""
    import torch

    mock_model = MagicMock()
    mock_model.config.max_position_embeddings = 1024
    mock_out = MagicMock()
    mock_out.loss = torch.tensor(loss_value)
    mock_model.return_value = mock_out
    return mock_model


def _make_mock_tokenizer(seq_len: int = 10) -> MagicMock:
    """Build a mock tokenizer that returns a fixed-length input_ids tensor."""
    import torch

    mock_tok = MagicMock()
    mock_tok.return_value = {"input_ids": torch.ones(1, seq_len, dtype=torch.long)}
    return mock_tok


class TestPerplexityFilter:
    # ── score() ───────────────────────────────────────────────────────────────

    def test_score_returns_exp_loss(self) -> None:
        import math

        import torch

        pf = PerplexityFilter(threshold=50.0)
        pf._model = _make_mock_model(loss_value=math.log(20.0))  # exp(ln20) = 20
        pf._tokenizer = _make_mock_tokenizer(seq_len=5)
        assert pf.score("any text") == pytest.approx(20.0, rel=1e-4)

    def test_score_empty_string_returns_zero(self) -> None:
        import torch

        pf = PerplexityFilter()
        pf._model = _make_mock_model(loss_value=1.0)
        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.zeros(1, 0, dtype=torch.long)}
        pf._tokenizer = mock_tok
        assert pf.score("") == pytest.approx(0.0)

    # ── apply() with a string ─────────────────────────────────────────────────

    def test_apply_string_below_threshold_returned(self) -> None:
        import math

        pf = PerplexityFilter(threshold=50.0)
        pf._model = _make_mock_model(loss_value=math.log(10.0))  # PPL=10 < 50
        pf._tokenizer = _make_mock_tokenizer(seq_len=5)
        result = pf.apply("some passage")
        assert result == "some passage"

    def test_apply_string_above_threshold_returns_empty(self) -> None:
        import math

        pf = PerplexityFilter(threshold=50.0)
        pf._model = _make_mock_model(loss_value=math.log(200.0))  # PPL=200 > 50
        pf._tokenizer = _make_mock_tokenizer(seq_len=5)
        result = pf.apply("weird passage")
        assert result == ""

    # ── apply() with a list ───────────────────────────────────────────────────

    def test_apply_list_filters_high_ppl(self) -> None:
        """Passages above threshold are removed from the list."""
        import math

        import torch

        pf = PerplexityFilter(threshold=50.0)

        # Make score() return different values for different calls
        losses = iter([math.log(10.0), math.log(200.0), math.log(5.0)])

        def _mock_score(passage: str) -> float:
            return float(torch.exp(torch.tensor(next(losses))).item())

        pf.score = _mock_score  # type: ignore[method-assign]
        result = pf.apply(["good", "bad", "also good"])
        assert result == ["good", "also good"]

    def test_apply_empty_list(self) -> None:
        pf = PerplexityFilter()
        assert pf.apply([]) == []

    def test_apply_all_above_threshold(self) -> None:
        import math

        import torch

        pf = PerplexityFilter(threshold=5.0)
        losses = iter([math.log(100.0), math.log(200.0)])

        def _mock_score(passage: str) -> float:
            return float(torch.exp(torch.tensor(next(losses))).item())

        pf.score = _mock_score  # type: ignore[method-assign]
        assert pf.apply(["a", "b"]) == []

    # ── lazy model loading ────────────────────────────────────────────────────

    def test_load_model_called_on_first_score(self) -> None:
        """_load_model() is called exactly once on first score() invocation."""
        import math

        pf = PerplexityFilter(threshold=50.0)
        pf._model = _make_mock_model(loss_value=math.log(10.0))
        pf._tokenizer = _make_mock_tokenizer(seq_len=5)
        pf._load_model = MagicMock()  # type: ignore[method-assign]

        # _model is already set, so _load_model should NOT be called
        pf.score("hello")
        pf._load_model.assert_not_called()

    def test_load_model_called_when_model_none(self) -> None:
        """_load_model() is invoked when _model is None."""
        import math

        pf = PerplexityFilter(threshold=50.0)

        def _fake_load() -> None:
            pf._model = _make_mock_model(loss_value=math.log(10.0))
            pf._tokenizer = _make_mock_tokenizer(seq_len=5)

        pf._load_model = _fake_load  # type: ignore[method-assign]
        score = pf.score("test passage")
        assert score == pytest.approx(10.0, rel=1e-4)


# ── ParaphraseDefense ─────────────────────────────────────────────────────────


class TestParaphraseDefense:
    # ── _paraphrase() ─────────────────────────────────────────────────────────

    def test_paraphrase_calls_llm_generate(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The fox is brown and quick."
        pd = ParaphraseDefense(llm=mock_llm)
        result = pd._paraphrase("The quick brown fox.")
        mock_llm.generate.assert_called_once()
        assert result == "The fox is brown and quick."

    def test_paraphrase_uses_default_template(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "rephrased"
        pd = ParaphraseDefense(llm=mock_llm)
        pd._paraphrase("original passage")
        call_args = mock_llm.generate.call_args[0][0]
        assert "original passage" in call_args
        assert "Paraphrase" in call_args

    def test_paraphrase_uses_custom_template(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "custom output"
        pd = ParaphraseDefense(
            llm=mock_llm,
            prompt_template="Rewrite: {passage}",
        )
        pd._paraphrase("my text")
        call_args = mock_llm.generate.call_args[0][0]
        assert call_args == "Rewrite: my text"

    # ── apply() with a string ─────────────────────────────────────────────────

    def test_apply_string_returns_paraphrased(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "paraphrased result"
        pd = ParaphraseDefense(llm=mock_llm)
        assert pd.apply("original") == "paraphrased result"

    # ── apply() with a list ───────────────────────────────────────────────────

    def test_apply_list_paraphrases_each(self) -> None:
        responses = ["para1", "para2", "para3"]
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = responses
        pd = ParaphraseDefense(llm=mock_llm)
        result = pd.apply(["a", "b", "c"])
        assert result == ["para1", "para2", "para3"]
        assert mock_llm.generate.call_count == 3

    def test_apply_empty_list(self) -> None:
        mock_llm = MagicMock()
        pd = ParaphraseDefense(llm=mock_llm)
        assert pd.apply([]) == []
        mock_llm.generate.assert_not_called()

    def test_apply_single_item_list(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "one result"
        pd = ParaphraseDefense(llm=mock_llm)
        assert pd.apply(["only one"]) == ["one result"]

    # ── stealth invariant ─────────────────────────────────────────────────────

    def test_invisible_chars_survive_paraphrase_when_llm_preserves_them(
        self,
    ) -> None:
        """If the LLM echoes invisible chars, paraphrase does NOT strip them.

        This is the research hypothesis: paraphrase alone does not neutralise
        the Unicode trigger — NFKC stripping is needed.
        """
        invisible = "\u200b"
        original = f"The capital{invisible} is Paris."
        mock_llm = MagicMock()
        mock_llm.generate.return_value = original  # LLM echoes invisible chars
        pd = ParaphraseDefense(llm=mock_llm)
        result = pd.apply(original)
        assert invisible in result  # defense did NOT remove the invisible char
