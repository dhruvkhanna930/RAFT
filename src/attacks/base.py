"""Abstract base class for all RAG corpus-poisoning attacks.

Every concrete attack must subclass :class:`AttackBase` and implement:

- :meth:`craft_malicious_passages` — given a target question and desired
  answer, return *n* adversarial passage strings ready for corpus injection.
- :meth:`inject` — insert a list of adversarial passage strings into a corpus.

Keeping these two operations separate lets experiment scripts reuse crafted
passages across different corpora, or inject at different points in time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Shared data types ─────────────────────────────────────────────────────────


@dataclass
class AttackConfig:
    """Lightweight config passed to every :class:`AttackBase` subclass.

    Populated from ``configs/attacks.yaml`` via
    :func:`src.utils.config.load_project_config`.

    Args:
        injection_budget: Maximum adversarial passages injected per question.
        extra: Attack-specific hyper-parameters forwarded from the YAML.
    """

    injection_budget: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdversarialPassage:
    """Metadata container for a single crafted adversarial passage.

    The ``text`` field is what gets injected into the corpus.  The remaining
    fields are kept for logging, ablations, and stealth metrics — they are
    never passed through the :class:`AttackBase` interface.

    Args:
        text: Passage string (may contain invisible Unicode characters).
        target_answer: Answer the attack aims to elicit from the LLM.
        source_question: Target question this passage was crafted for.
        metadata: Attack-specific extras (loss values, perturbation counts, …).
    """

    text: str
    target_answer: str
    source_question: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Abstract base ─────────────────────────────────────────────────────────────


class AttackBase(ABC):
    """Abstract base class for RAG corpus-poisoning attacks.

    Subclasses must implement :meth:`craft_malicious_passages` and
    :meth:`inject`.  All other methods are optional helpers.

    Args:
        config: :class:`AttackConfig` with shared hyper-parameters.
    """

    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    @abstractmethod
    def craft_malicious_passages(
        self,
        target_question: str,
        target_answer: str,
        n: int,
    ) -> list[str]:
        """Craft *n* adversarial passage strings for the given target pair.

        This is the primary interface used by experiment scripts.  Returns
        plain strings so the caller does not depend on any internal data
        structure.  Concrete implementations may build :class:`AdversarialPassage`
        objects internally for metadata tracking but must expose plain text here.

        Args:
            target_question: The question the attack is designed to answer
                incorrectly.
            target_answer: The malicious answer the LLM should produce.
            n: Number of adversarial passages to craft.  Callers typically
                pass ``self.config.injection_budget``.

        Returns:
            List of *n* adversarial passage strings.
        """

    @abstractmethod
    def inject(
        self,
        corpus: list[str],
        adversarial_passages: list[str],
    ) -> list[str]:
        """Insert *adversarial_passages* into *corpus*.

        Args:
            corpus: Original passage list (not modified in-place).
            adversarial_passages: Strings returned by
                :meth:`craft_malicious_passages`.

        Returns:
            New list containing all original passages plus the adversarial
            ones, in implementation-defined positions.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(budget={self.config.injection_budget})"


# Backwards-compatibility alias — remove once all imports are updated.
BaseAttack = AttackBase
