"""Tests for src/attacks/unicode_chars — perturb() and UnicodeInventory.

Split into two suites:

1. ``TestPerturb`` — pure-Python tests for the ``perturb()`` function.
   No ML models required; instant with ``pytest``.

2. ``TestContrieverEmbeddingChange`` — integration tests that load
   ``facebook/contriever`` and verify that Unicode insertions change the
   resulting embedding.  Marked ``@pytest.mark.slow``; skipped by default.
   Run with::

       pytest -m slow tests/test_unicode_chars.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks.unicode_chars import (
    CharCategory,
    UnicodeInventory,
    _ALL_INVISIBLE,
    perturb,
)


# ── perturb() ─────────────────────────────────────────────────────────────────


class TestPerturb:
    def test_length_increases_by_insertion_count(self) -> None:
        text = "hello world"
        insertions = [(5, "\u200b"), (2, "\u200c")]
        result = perturb(text, insertions)
        assert len(result) == len(text) + 2

    def test_visual_identity_via_strip(self) -> None:
        """strip_invisible on the perturbed text returns the original."""
        text = "hello world"
        insertions = [(5, "\u200b"), (2, "\u200c")]
        result = perturb(text, insertions)
        assert UnicodeInventory.strip_invisible(result) == text

    def test_empty_insertions_returns_original(self) -> None:
        text = "hello"
        assert perturb(text, []) == text

    def test_single_insertion_at_start(self) -> None:
        result = perturb("abc", [(0, "\u200b")])
        assert result == "\u200babc"
        assert len(result) == 4

    def test_single_insertion_at_end(self) -> None:
        result = perturb("abc", [(3, "\u200b")])
        assert result == "abc\u200b"
        assert len(result) == 4

    def test_right_to_left_ordering(self) -> None:
        """Two insertions at distinct positions must both land correctly."""
        # Insert "Y" at pos 1, "X" at pos 2 → right-to-left: X first.
        # After inserting X at 2: "abXc"
        # After inserting Y at 1: "aYbXc"
        result = perturb("abc", [(2, "X"), (1, "Y")])
        assert result == "aYbXc"
        assert len(result) == 5

    def test_all_invisible_roundtrip(self) -> None:
        """Inserting 5 invisible chars and stripping recovers the original."""
        text = "the quick brown fox"
        inv = UnicodeInventory()
        insertions = [(i * 2, inv.chars[i % len(inv)]) for i in range(5)]
        perturbed = perturb(text, insertions)
        assert UnicodeInventory.strip_invisible(perturbed) == text

    def test_position_clamped_beyond_length(self) -> None:
        """Positions > len(text) must be clamped to the end."""
        result = perturb("abc", [(999, "\u200b")])
        assert result == "abc\u200b"

    def test_empty_text_single_insertion(self) -> None:
        result = perturb("", [(0, "\u200b")])
        assert result == "\u200b"

    def test_empty_text_no_insertion(self) -> None:
        assert perturb("", []) == ""

    def test_negative_position_clamped_to_zero(self) -> None:
        result = perturb("abc", [(-5, "\u200b")])
        assert result == "\u200babc"

    def test_tag_char_invisible_to_strip(self) -> None:
        tag = chr(0xE0061)  # tag 'a'
        text = "hello world"
        perturbed = perturb(text, [(5, tag)])
        assert tag not in UnicodeInventory.strip_invisible(perturbed)
        assert UnicodeInventory.strip_invisible(perturbed) == text

    def test_many_insertions_roundtrip(self) -> None:
        """50 invisible chars inserted and stripped back to original."""
        text = "The quick brown fox jumps over the lazy dog."
        inv = UnicodeInventory()
        insertions = [
            (i % (len(text) + 1), inv.chars[i % len(inv)]) for i in range(50)
        ]
        perturbed = perturb(text, insertions)
        assert len(perturbed) == len(text) + 50
        assert UnicodeInventory.strip_invisible(perturbed) == text

    def test_visible_chars_unaffected(self) -> None:
        """All original visible characters are present in the same order."""
        text = "abcdef"
        insertions = [(1, "\u200b"), (3, "\u200c"), (5, "\u200d")]
        result = perturb(text, insertions)
        visible = "".join(c for c in result if c not in _ALL_INVISIBLE)
        assert visible == text


# ── Integration: Contriever embedding changes after perturbation ──────────────


@pytest.mark.slow
class TestContrieverEmbeddingChange:
    """Load facebook/contriever and verify that Unicode perturbations
    produce different embedding vectors.

    These tests download the model on first run (~440 MB).
    """

    @pytest.fixture(scope="class")
    def retriever(self):  # type: ignore[no-untyped-def]
        """Return a ContrieverRetriever with the model already loaded."""
        from src.retrievers.contriever import ContrieverRetriever

        r = ContrieverRetriever(device="auto", batch_size=32, normalize=True)
        r._load_model()
        return r

    def test_massive_insertion_changes_embedding(self, retriever) -> None:  # type: ignore[no-untyped-def]
        """30 invisible chars spread through the text must shift the embedding."""
        inv = UnicodeInventory()
        text = "Paris is the capital of France."
        insertions = [
            (i * 2 % (len(text) + 1), inv.chars[i % len(inv)]) for i in range(30)
        ]
        perturbed = perturb(text, insertions)

        emb_clean = retriever.embed([text])[0]
        emb_perturbed = retriever.embed([perturbed])[0]
        cos_sim = float(np.dot(emb_clean, emb_perturbed))

        # If the tokeniser strips all invisible chars the embeddings are
        # identical — that is itself a research finding worth surfacing.
        assert cos_sim < 1.0 - 1e-6, (
            f"Embeddings unchanged after perturbation (cos_sim={cos_sim:.8f}). "
            "The Contriever tokeniser may be stripping all Cf/Mn characters — "
            "see research notes on BERT normalisation."
        )

    def test_variation_selector_changes_embedding(self, retriever) -> None:  # type: ignore[no-untyped-def]
        """Variation selectors (Unicode Mn category) should survive BERT tokenisation."""
        inv = UnicodeInventory(categories=[CharCategory.VARIATION_SELECTORS])
        if len(inv) == 0:
            pytest.skip("No variation selectors in inventory")

        text = "The answer is London."
        # Insert the selector after the first character.
        perturbed = perturb(text, [(1, inv.chars[0])])

        emb_clean = retriever.embed([text])[0]
        emb_perturbed = retriever.embed([perturbed])[0]
        cos_sim = float(np.dot(emb_clean, emb_perturbed))

        assert cos_sim < 1.0 - 1e-6, (
            f"Variation selector did not change embedding (cos_sim={cos_sim:.8f}). "
            "Tokeniser may strip Mn characters when strip_accents=True."
        )

    def test_baseline_clean_similarity_is_one(self, retriever) -> None:
        """Sanity: embedding a text twice gives cosine sim ≈ 1.0."""
        text = "hello world"
        e1 = retriever.embed([text])[0]
        e2 = retriever.embed([text])[0]
        assert float(np.dot(e1, e2)) == pytest.approx(1.0, abs=1e-5)
