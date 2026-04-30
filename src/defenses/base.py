"""Abstract base class for all post-retrieval and corpus-level defenses.

Every concrete defense must subclass :class:`DefenseBase` and implement
:meth:`apply`.  The method accepts either a single passage string or a list,
so defenses compose naturally over both individual passages and retrieval sets.

Calling convention::

    normalizer = UnicodeNormalizer()

    # Single passage (e.g. inside a generation loop):
    clean = normalizer.apply("hello\\u200bworld")        # → "helloworld"

    # Batch (e.g. applied to full retrieved context):
    clean = normalizer.apply(["hello\\u200bworld", ...]) # → ["helloworld", ...]
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DefenseBase(ABC):
    """Abstract base class for text / passage defenses.

    A defense transforms one or more passage strings — stripping adversarial
    content, filtering low-quality passages, or rewriting text — without
    any knowledge of the downstream RAG system or corpus size.

    Subclasses must implement :meth:`apply`.
    """

    @abstractmethod
    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Apply the defense to a single passage or a list of passages.

        Implementations must preserve the input type: a ``str`` input must
        return a ``str``; a ``list[str]`` input must return a ``list[str]``.

        Args:
            text_or_passages: A single passage string, or a list of passage
                strings (e.g. the top-k retrieved context).

        Returns:
            Defended text or list of defended texts, matching the input type.
        """

    def __call__(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Alias for :meth:`apply` so defenses work as callables."""
        return self.apply(text_or_passages)

    def then(self, other: "DefenseBase") -> "DefenseChain":
        """Chain this defense with *other*, returning a :class:`DefenseChain`.

        Example::

            clean = UnicodeNormalizer().then(DuplicateFilter()).apply(passages)

        Args:
            other: The next defense to apply after this one.

        Returns:
            A :class:`DefenseChain` that applies ``self`` then ``other``.
        """
        return DefenseChain([self, other])


class DefenseChain(DefenseBase):
    """Apply a sequence of defenses in order.

    Defenses are applied left-to-right: the output of each becomes the
    input of the next.  All defenses must satisfy :class:`DefenseBase`.

    Args:
        defenses: Ordered list of :class:`DefenseBase` instances.

    Example::

        chain = DefenseChain([UnicodeNormalizer(), DuplicateFilter()])
        clean_passages = chain.apply(retrieved_passages)
    """

    def __init__(self, defenses: list[DefenseBase]) -> None:
        if not defenses:
            raise ValueError("DefenseChain requires at least one defense.")
        self.defenses = list(defenses)

    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        """Apply all defenses sequentially.

        Args:
            text_or_passages: Input passage(s).

        Returns:
            Output after all defenses have been applied in order.
        """
        result: str | list[str] = text_or_passages
        for defense in self.defenses:
            result = defense.apply(result)
        return result

    def then(self, other: DefenseBase) -> "DefenseChain":
        """Extend the chain with one more defense.

        Args:
            other: Defense to append.

        Returns:
            New :class:`DefenseChain` with *other* appended.
        """
        return DefenseChain(self.defenses + [other])
