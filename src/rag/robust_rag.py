"""RobustRAG variant (Xiang et al., USENIX Security 2024).

Certified robustness via isolate-then-aggregate: each retrieved passage is
fed to the LLM in isolation; final answer is determined by majority vote over
keyword-extracted per-passage answers.

**Deviation from original:** The upstream repo uses GPT-4 with a specific
prompt format.  We use whatever LLM is passed to the constructor (same as
the other RAG variants).  Keyword extraction uses a built-in English stopword
set (~174 words) rather than an NLTK dependency at runtime.

Upstream repo: https://github.com/inspire-group/RobustRAG
Pins: torch 2.2.1, transformers 4.40.1.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from src.rag.base import GenerationResult, RagBase, RetrievalResult

logger = logging.getLogger(__name__)

_ISOLATED_PROMPT = """\
Answer the question based on the given passage. \
Only give me the answer and do not output any other words.

Passage: {passage}

Answer the question based on the given passage. \
Only give me the answer and do not output any other words.
Question: {question}
Answer:"""

# Hardcoded English stopwords — avoids nltk.download() at runtime.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "let's", "may", "me", "might",
    "more", "most", "must", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "say", "says",
    "sha", "shan't", "she", "should", "shouldn't", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "upon", "us", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "won't", "would", "wouldn't", "yes",
    "yet", "you", "your", "yours", "yourself", "yourselves",
})


class RobustRAG(RagBase):
    """Certified-robust RAG via per-passage isolation and majority vote.

    Each passage produces an independent LLM generation.  The final answer
    is selected by majority vote across the isolated answers, ensuring that
    a single poisoned passage cannot dominate the output.

    Args:
        retriever: Base retriever.
        llm: LLM client (called ``top_k`` times per query — one per passage).
        top_k: Number of passages to retrieve (each processed in isolation).
        aggregation: Aggregation strategy — ``"majority_vote"`` (default) or
            ``"keyword_intersection"``.
        keyword_extractor: Optional callable ``(str) -> set[str]``.  If
            ``None``, :meth:`_extract_keywords` is used.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        top_k: int = 5,
        aggregation: str = "majority_vote",
        keyword_extractor: Any = None,
    ) -> None:
        super().__init__(retriever, llm, top_k)
        self.aggregation = aggregation
        self.keyword_extractor = keyword_extractor

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        """Standard retrieval — RobustRAG's defence is in generate().

        Args:
            query: User question.
            k: Number of passages to retrieve.

        Returns:
            ``RetrievalResult`` with top-k passages.
        """
        passages, scores = self.retriever.retrieve(query, k=k)
        return RetrievalResult(passages=passages, scores=scores)

    def generate(self, query: str, retrieved: RetrievalResult) -> GenerationResult:
        """Generate one answer per passage in isolation, then aggregate.

        Each passage is used as the sole context for an independent LLM
        call.  The isolated answers are then aggregated via majority vote
        (or keyword intersection) to produce the final answer.

        Args:
            query: User question.
            retrieved: ``RetrievalResult`` with top-k passages.

        Returns:
            ``GenerationResult`` with majority-voted answer and per-passage
            isolated generations in metadata.
        """
        isolated_answers: list[str] = []
        for passage in retrieved.passages:
            prompt = _ISOLATED_PROMPT.format(passage=passage, question=query)
            ans = self.llm.generate(prompt).strip()
            isolated_answers.append(ans)

        final_answer = self._aggregate(isolated_answers)

        return GenerationResult(
            answer=final_answer,
            retrieved=retrieved,
            metadata={
                "isolated_answers": isolated_answers,
                "aggregation": self.aggregation,
            },
        )

    def _aggregate(self, isolated_answers: list[str]) -> str:
        """Aggregate per-passage answers into the final answer.

        ``"majority_vote"``: most common full answer string wins.
        ``"keyword_intersection"``: keep keywords appearing in >= ceil(k/2)
        answers; return the isolated answer with highest keyword overlap.

        Args:
            isolated_answers: Per-passage LLM outputs.

        Returns:
            Aggregated answer string.
        """
        if not isolated_answers:
            return ""

        if self.aggregation == "keyword_intersection":
            return self._keyword_aggregate(isolated_answers)

        # Default: majority_vote — most common normalised answer.
        normalised = [a.strip().lower() for a in isolated_answers]
        counts = Counter(normalised)
        winner = counts.most_common(1)[0][0]
        # Return the original-cased version of the winner.
        for ans in isolated_answers:
            if ans.strip().lower() == winner:
                return ans.strip()
        return isolated_answers[0].strip()

    def _keyword_aggregate(self, isolated_answers: list[str]) -> str:
        """Keyword-intersection aggregation strategy.

        1. Extract keywords from each isolated answer.
        2. Count each keyword across all answers.
        3. Keep keywords appearing in >= ceil(k/2) answers.
        4. Return the isolated answer whose keywords have the highest
           overlap with the surviving keyword set.

        Args:
            isolated_answers: Per-passage LLM outputs.

        Returns:
            Best-matching isolated answer string.
        """
        extractor = self.keyword_extractor or self._extract_keywords
        per_answer_kws: list[set[str]] = [
            extractor(a) for a in isolated_answers
        ]

        # Count keyword frequency across answers.
        kw_counter: Counter[str] = Counter()
        for kws in per_answer_kws:
            for kw in kws:
                kw_counter[kw] += 1

        threshold = math.ceil(len(isolated_answers) / 2)
        surviving_kws = {kw for kw, c in kw_counter.items() if c >= threshold}

        if not surviving_kws:
            # Fallback: return most common full answer.
            normalised = [a.strip().lower() for a in isolated_answers]
            counts = Counter(normalised)
            winner = counts.most_common(1)[0][0]
            for ans in isolated_answers:
                if ans.strip().lower() == winner:
                    return ans.strip()
            return isolated_answers[0].strip()

        # Pick the isolated answer with the best overlap.
        best_idx = 0
        best_overlap = 0
        for idx, kws in enumerate(per_answer_kws):
            overlap = len(kws & surviving_kws)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        return isolated_answers[best_idx].strip()

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract content keywords from an answer string.

        Lowercases, splits on non-alphanumeric boundaries, and removes
        stopwords.

        Args:
            text: LLM output string.

        Returns:
            Set of lowercase content word strings.
        """
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 1}
