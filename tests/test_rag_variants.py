"""Tests for advanced RAG variants — SelfRAG, CRAG, TrustRAG, RobustRAG.

All tests use mocked retriever + LLM; no ML models required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.rag.base import GenerationResult, RetrievalResult


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _mock_retriever(
    passages: list[str] | None = None,
    scores: list[float] | None = None,
    embed_dim: int = 768,
) -> MagicMock:
    """Build a mock retriever returning canned passages/scores."""
    if passages is None:
        passages = [
            "Paris is the capital of France.",
            "France is a country in Europe.",
            "The Eiffel Tower is in Paris.",
            "Berlin is the capital of Germany.",
            "London is the capital of the UK.",
        ]
    if scores is None:
        scores = [0.95, 0.85, 0.75, 0.60, 0.50]

    retriever = MagicMock()
    retriever.retrieve.return_value = (passages, scores)
    # For TrustRAG: embed() returns random fixed vectors.
    rng = np.random.RandomState(42)
    retriever.embed.return_value = rng.randn(len(passages), embed_dim).tolist()
    return retriever


def _mock_llm(answer: str = "Paris") -> MagicMock:
    """Build a mock LLM that always returns a fixed answer."""
    llm = MagicMock()
    llm.generate.return_value = answer
    return llm


# ── SelfRAG ──────────────────────────────────────────────────────────────────

class TestSelfRAG:
    def test_answer_returns_generation_result(self) -> None:
        from src.rag.self_rag import SelfRAG

        # side_effect: 5 "Yes" relevance checks, then the final answer
        llm = MagicMock()
        llm.generate.side_effect = ["Yes", "Yes", "No", "Yes", "No", "Paris"]
        rag = SelfRAG(retriever=_mock_retriever(), llm=llm, top_k=5)

        result = rag.answer("What is the capital of France?")
        assert isinstance(result, GenerationResult)
        assert isinstance(result.retrieved, RetrievalResult)
        assert result.answer == "Paris"

    def test_relevance_filtering(self) -> None:
        from src.rag.self_rag import SelfRAG

        llm = MagicMock()
        # Only passage 0 and 2 are relevant
        llm.generate.side_effect = ["Yes", "No", "Yes", "No", "No", "Paris"]
        rag = SelfRAG(retriever=_mock_retriever(), llm=llm, top_k=5)

        result = rag.answer("What is the capital of France?")
        # Should have filtered to 2 relevant passages
        assert len(result.retrieved.passages) == 2
        assert result.metadata["relevance"] == [True, False, True, False, False]

    def test_fallback_when_none_relevant(self) -> None:
        from src.rag.self_rag import SelfRAG

        llm = MagicMock()
        # All "No" relevance checks → fallback to all passages
        llm.generate.side_effect = ["No", "No", "No", "No", "No", "Paris"]
        rag = SelfRAG(retriever=_mock_retriever(), llm=llm, top_k=5)

        result = rag.answer("What is the capital of France?")
        # Fallback: all 5 passages used
        assert len(result.retrieved.passages) == 5
        assert result.answer == "Paris"

    def test_retrieve_delegates_to_retriever(self) -> None:
        from src.rag.self_rag import SelfRAG

        retriever = _mock_retriever()
        rag = SelfRAG(retriever=retriever, llm=_mock_llm(), top_k=5)

        result = rag.retrieve("test query", k=3)
        retriever.retrieve.assert_called_once_with("test query", k=3)
        assert isinstance(result, RetrievalResult)


# ── CRAG ─────────────────────────────────────────────────────────────────────

class TestCRAG:
    def test_answer_returns_generation_result(self) -> None:
        from src.rag.crag import CRAG

        rag = CRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm("Paris"),
            top_k=5,
            upper_threshold=0.7,
            lower_threshold=0.3,
        )
        result = rag.answer("What is the capital of France?")
        assert isinstance(result, GenerationResult)
        assert result.answer == "Paris"

    def test_route_correct(self) -> None:
        from src.rag.crag import CRAG, RetrievalAction

        rag = CRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            upper_threshold=0.8,
            lower_threshold=0.4,
        )
        assert rag._route(0.9) == RetrievalAction.CORRECT
        assert rag._route(0.8) == RetrievalAction.CORRECT

    def test_route_incorrect(self) -> None:
        from src.rag.crag import CRAG, RetrievalAction

        rag = CRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            upper_threshold=0.8,
            lower_threshold=0.4,
        )
        assert rag._route(0.3) == RetrievalAction.INCORRECT
        assert rag._route(0.4) == RetrievalAction.INCORRECT

    def test_route_ambiguous(self) -> None:
        from src.rag.crag import CRAG, RetrievalAction

        rag = CRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            upper_threshold=0.8,
            lower_threshold=0.4,
        )
        assert rag._route(0.5) == RetrievalAction.AMBIGUOUS
        assert rag._route(0.6) == RetrievalAction.AMBIGUOUS
        assert rag._route(0.79) == RetrievalAction.AMBIGUOUS

    def test_filtering_keeps_correct_discards_incorrect(self) -> None:
        from src.rag.crag import CRAG

        # Scores: 0.95, 0.85, 0.75, 0.60, 0.50
        # With thresholds 0.8/0.4: CORRECT, CORRECT, AMBIGUOUS, AMBIGUOUS, AMBIGUOUS
        llm = MagicMock()
        llm.generate.side_effect = [
            # 3 refined queries for AMBIGUOUS passages
            "refined query 1",
            "refined query 2",
            "refined query 3",
            # Final answer generation
            "Paris",
        ]
        retriever = _mock_retriever()
        # Re-retrieval for refined queries returns a single passage
        retriever.retrieve.side_effect = [
            # Initial retrieval
            (
                ["p1", "p2", "p3", "p4", "p5"],
                [0.95, 0.85, 0.75, 0.60, 0.50],
            ),
            # 3 re-retrievals for AMBIGUOUS passages
            (["re-retrieved 1"], [0.9]),
            (["re-retrieved 2"], [0.7]),
            (["re-retrieved 3"], [0.6]),
        ]

        rag = CRAG(
            retriever=retriever,
            llm=llm,
            top_k=5,
            upper_threshold=0.8,
            lower_threshold=0.4,
        )
        result = rag.retrieve("What is the capital of France?", k=5)
        # 2 CORRECT + 3 re-retrieved from AMBIGUOUS = 5
        assert len(result.passages) == 5
        assert "routing" in result.metadata

    def test_refine_query_calls_llm(self) -> None:
        from src.rag.crag import CRAG

        llm = MagicMock()
        llm.generate.return_value = "What specific city is the capital of France?"
        rag = CRAG(retriever=_mock_retriever(), llm=llm)
        refined = rag._refine_query("capital of France?", "Paris is a city.")
        assert len(refined) > 5
        llm.generate.assert_called_once()

    def test_refine_query_fallback_on_empty(self) -> None:
        from src.rag.crag import CRAG

        llm = MagicMock()
        llm.generate.return_value = ""
        rag = CRAG(retriever=_mock_retriever(), llm=llm)
        refined = rag._refine_query("original question", "some passage")
        assert refined == "original question"

    def test_fallback_when_all_discarded(self) -> None:
        from src.rag.crag import CRAG

        # All scores below lower_threshold → all INCORRECT
        retriever = _mock_retriever(
            passages=["p1", "p2"],
            scores=[0.1, 0.2],
        )
        rag = CRAG(
            retriever=retriever,
            llm=_mock_llm("Paris"),
            top_k=2,
            upper_threshold=0.8,
            lower_threshold=0.4,
        )
        result = rag.retrieve("test?", k=2)
        # Fallback: top-1 passage kept
        assert len(result.passages) == 1
        assert result.passages[0] == "p1"


# ── TrustRAG ────────────────────────────────────────────────────────────────

class TestTrustRAG:
    def test_answer_returns_generation_result(self) -> None:
        from src.rag.trust_rag import TrustRAG

        rag = TrustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm("Paris"),
            top_k=3,
            retrieve_k=5,
            n_clusters=2,
        )
        result = rag.answer("What is the capital of France?")
        assert isinstance(result, GenerationResult)
        assert result.answer == "Paris"

    def test_kmeans_filter_majority_cluster(self) -> None:
        from src.rag.trust_rag import TrustRAG

        rag = TrustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            n_clusters=2,
        )

        # Create embeddings: 4 similar + 1 outlier
        rng = np.random.RandomState(42)
        majority = rng.randn(4, 10) + 5.0  # cluster around 5
        minority = rng.randn(1, 10) - 5.0  # cluster around -5
        embeddings = np.vstack([majority, minority])

        passages = ["p1", "p2", "p3", "p4", "outlier"]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]

        filtered_p, filtered_s, meta = rag._kmeans_filter(
            embeddings, passages, scores
        )

        # The outlier should be in the minority cluster and removed.
        assert "outlier" not in filtered_p
        assert len(filtered_p) == 4
        assert meta["n_discarded"] == 1
        assert meta["n_kept"] == 4

    def test_kmeans_skipped_when_few_passages(self) -> None:
        from src.rag.trust_rag import TrustRAG

        retriever = _mock_retriever(passages=["p1"], scores=[0.9])
        rag = TrustRAG(
            retriever=retriever,
            llm=_mock_llm(),
            top_k=1,
            retrieve_k=1,
            n_clusters=2,
        )
        result = rag.retrieve("test?", k=1)
        assert result.metadata.get("kmeans_skipped") is True
        assert len(result.passages) == 1

    def test_truncates_to_top_k(self) -> None:
        from src.rag.trust_rag import TrustRAG

        # 5 passages, all in one cluster (embeddings are similar enough)
        retriever = _mock_retriever()
        # Override embed to return very similar embeddings (same cluster)
        retriever.embed.return_value = np.ones((5, 10)).tolist()
        rag = TrustRAG(
            retriever=retriever,
            llm=_mock_llm(),
            top_k=2,
            retrieve_k=5,
            n_clusters=2,
        )
        result = rag.retrieve("test?", k=2)
        assert len(result.passages) <= 2


# ── RobustRAG ────────────────────────────────────────────────────────────────

class TestRobustRAG:
    def test_answer_returns_generation_result(self) -> None:
        from src.rag.robust_rag import RobustRAG

        llm = MagicMock()
        # 5 isolated answers + no extra call needed for majority vote
        llm.generate.side_effect = ["Paris", "Paris", "London", "Paris", "Berlin"]
        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=llm,
            top_k=5,
            aggregation="majority_vote",
        )
        result = rag.answer("What is the capital of France?")
        assert isinstance(result, GenerationResult)
        assert result.answer == "Paris"  # majority = Paris (3/5)
        assert "isolated_answers" in result.metadata
        assert len(result.metadata["isolated_answers"]) == 5

    def test_majority_vote_picks_most_common(self) -> None:
        from src.rag.robust_rag import RobustRAG

        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            aggregation="majority_vote",
        )
        winner = rag._aggregate(["Paris", "London", "Paris", "Berlin", "Paris"])
        assert winner == "Paris"

    def test_majority_vote_case_insensitive(self) -> None:
        from src.rag.robust_rag import RobustRAG

        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            aggregation="majority_vote",
        )
        winner = rag._aggregate(["paris", "Paris", "PARIS"])
        assert winner.lower() == "paris"

    def test_extract_keywords_removes_stopwords(self) -> None:
        from src.rag.robust_rag import RobustRAG

        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
        )
        kws = rag._extract_keywords("The capital of France is Paris")
        assert "capital" in kws
        assert "france" in kws
        assert "paris" in kws
        # Stopwords removed
        assert "the" not in kws
        assert "of" not in kws
        assert "is" not in kws

    def test_extract_keywords_empty_string(self) -> None:
        from src.rag.robust_rag import RobustRAG

        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
        )
        assert rag._extract_keywords("") == set()

    def test_keyword_intersection_aggregation(self) -> None:
        from src.rag.robust_rag import RobustRAG

        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
            aggregation="keyword_intersection",
        )
        answers = [
            "The capital of France is Paris",
            "Paris is the capital city",
            "London is the capital of the UK",
        ]
        result = rag._aggregate(answers)
        # "capital" and "paris" appear in 2/3 (>= ceil(3/2)=2)
        # Answer with most overlap should win
        assert isinstance(result, str)
        assert len(result) > 0

    def test_aggregate_empty_list(self) -> None:
        from src.rag.robust_rag import RobustRAG

        rag = RobustRAG(
            retriever=_mock_retriever(),
            llm=_mock_llm(),
        )
        assert rag._aggregate([]) == ""

    def test_retrieve_delegates_to_retriever(self) -> None:
        from src.rag.robust_rag import RobustRAG

        retriever = _mock_retriever()
        rag = RobustRAG(retriever=retriever, llm=_mock_llm(), top_k=5)

        result = rag.retrieve("test query", k=3)
        retriever.retrieve.assert_called_once_with("test query", k=3)
        assert isinstance(result, RetrievalResult)

    def test_isolated_generation_calls_llm_per_passage(self) -> None:
        from src.rag.robust_rag import RobustRAG

        passages = ["p1", "p2", "p3"]
        retriever = _mock_retriever(passages=passages, scores=[0.9, 0.8, 0.7])
        llm = MagicMock()
        llm.generate.side_effect = ["answer1", "answer1", "answer2"]

        rag = RobustRAG(retriever=retriever, llm=llm, top_k=3)
        retrieved = RetrievalResult(passages=passages, scores=[0.9, 0.8, 0.7])
        result = rag.generate("question?", retrieved)

        # LLM called once per passage (isolated)
        assert llm.generate.call_count == 3
        assert result.answer == "answer1"  # majority vote: 2/3
