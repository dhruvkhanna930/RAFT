"""Perplexity-based defense — filter high-PPL passages before generation.

Tests whether PPL filters catch Unicode-perturbed passages.
Hypothesis: invisible chars contribute near-zero perplexity under byte-level
or wordpiece tokenisers, so perturbed passages evade this defense.

Two filtering modes:

- **Threshold mode** (default): drop any passage with PPL > ``threshold``.
- **Top-fraction mode**: drop the highest-PPL ``drop_fraction`` of the
  retrieved set (e.g. 0.5 drops the worst half).  This is the mode used in
  the defense-grid experiment — it does not require tuning a PPL threshold.
"""

from __future__ import annotations

import math
from typing import Any

from src.defenses.base import DefenseBase


class PerplexityFilter(DefenseBase):
    """Drop passages by perplexity — threshold or top-fraction mode.

    Satisfies :class:`DefenseBase`: ``apply`` accepts a single string or a
    list.

    Args:
        scorer_model: HuggingFace model ID for the PPL scorer (default GPT-2).
        threshold: Passages with PPL above this value are removed
            (threshold mode).  Ignored when ``drop_fraction`` is set.
        drop_fraction: When set (0 < drop_fraction ≤ 1), drop the highest-PPL
            ``ceil(n * drop_fraction)`` passages from each retrieved batch
            instead of using the fixed threshold.  E.g. 0.5 drops the worst
            half (k/2 passages).
        device: Inference device for the scorer model (``"cpu"``, ``"mps"``,
            ``"cuda"``).
    """

    def __init__(
        self,
        scorer_model: str = "gpt2",
        threshold: float = 100.0,
        drop_fraction: float | None = None,
        device: str = "cpu",
    ) -> None:
        if drop_fraction is not None and not (0.0 < drop_fraction <= 1.0):
            raise ValueError(
                f"drop_fraction must be in (0, 1], got {drop_fraction}"
            )
        self.scorer_model = scorer_model
        self.threshold = threshold
        self.drop_fraction = drop_fraction
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    # ── DefenseBase interface ─────────────────────────────────────────────────

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Filter passage(s) by perplexity.

        Behaviour depends on ``drop_fraction``:

        - **Threshold mode** (``drop_fraction`` is ``None``): for a string,
          return it unchanged if PPL ≤ threshold, else return ``""``.  For a
          list, keep only passages with PPL ≤ threshold.
        - **Top-fraction mode** (``drop_fraction`` set): always operates on a
          list and drops the highest-PPL ``ceil(n × drop_fraction)`` passages.
          For a string, falls back to threshold mode.

        Args:
            text_or_passages: Single passage string, or a list.

        Returns:
            Filtered string or list (same type as input).
        """
        if isinstance(text_or_passages, str):
            return (
                text_or_passages
                if self.score(text_or_passages) <= self.threshold
                else ""
            )
        if self.drop_fraction is not None:
            return self._drop_top_fraction(text_or_passages)
        return [p for p in text_or_passages if self.score(p) <= self.threshold]

    # ── Top-fraction helper ───────────────────────────────────────────────────

    def _drop_top_fraction(self, passages: list[str]) -> list[str]:
        """Drop the highest-PPL ``ceil(n × drop_fraction)`` passages.

        Args:
            passages: Retrieved passage list (order preserved for survivors).

        Returns:
            Filtered list with the worst-PPL passages removed.
        """
        if not passages:
            return passages
        n_drop = math.ceil(len(passages) * self.drop_fraction)  # type: ignore[operator]
        scored = [(p, self.score(p)) for p in passages]
        scored.sort(key=lambda x: x[1], reverse=True)          # descending PPL
        drop_set = {id(p) for p, _ in scored[:n_drop]}
        # Preserve original order; drop by identity to handle duplicates safely.
        result: list[str] = []
        dropped = 0
        for p in passages:
            if id(p) in drop_set and dropped < n_drop:
                dropped += 1
            else:
                result.append(p)
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def score(self, passage: str) -> float:
        """Compute perplexity of *passage* under the scorer model.

        Uses cross-entropy loss from a causal LM forward pass: PPL = exp(loss).
        For passages longer than the model's maximum context, a stride-based
        approach is used to avoid truncation bias.

        Args:
            passage: Text to score.

        Returns:
            Perplexity float (lower = more fluent).
        """
        import torch

        if self._model is None:
            self._load_model()

        encodings = self._tokenizer(passage, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"].to(self.device)
        seq_len = input_ids.size(1)

        if seq_len == 0:
            return 0.0

        max_length: int = getattr(
            self._model.config, "max_position_embeddings", 1024
        )

        if seq_len <= max_length:
            with torch.no_grad():
                out = self._model(input_ids, labels=input_ids)
            return float(torch.exp(out.loss).item())

        # Stride-based PPL for passages longer than model context.
        stride = max_length // 2
        nlls: list[torch.Tensor] = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end
            chunk = input_ids[:, begin:end]
            labels = chunk.clone()
            labels[:, :-target_len] = -100  # mask prefix tokens
            with torch.no_grad():
                out = self._model(chunk, labels=labels)
            nlls.append(out.loss * target_len)
            prev_end = end
            if end >= seq_len:
                break

        return float(torch.exp(torch.stack(nlls).sum() / prev_end).item())

    def _load_model(self) -> None:
        """Lazily load the scorer model and tokenizer from HuggingFace.

        Called automatically on the first :meth:`score` call.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.scorer_model)
        self._model = AutoModelForCausalLM.from_pretrained(self.scorer_model)
        self._model.eval()
        if self.device != "cpu":
            self._model.to(self.device)
