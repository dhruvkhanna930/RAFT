"""Stealth / detectability metrics.

Measures how imperceptible the adversarial perturbation is:
- **PPL**: perplexity of the adversarial passage vs. a clean baseline.
- **Visual diff rate**: fraction of visibly altered glyphs (target: 0).
- **Char-class entropy**: fraction of non-ASCII / invisible characters.
"""

from __future__ import annotations

import difflib
import unicodedata

from src.attacks.unicode_chars import _ALL_INVISIBLE


def compute_stealth_metrics(
    adversarial_passages: list[str],
    clean_passages: list[str],
    ppl_scorer: object | None = None,
) -> dict[str, float]:
    """Compute all stealth metrics for a batch of adversarial passages.

    Args:
        adversarial_passages: Perturbed passage strings.
        clean_passages: Corresponding unperturbed passages (same content).
        ppl_scorer: Optional ``PerplexityFilter`` instance.  If None, PPL
            metrics are skipped and the PPL keys are absent from the output.

    Returns:
        Dict with keys: ``visual_diff_rate``, ``avg_zerowidth_frac``,
        ``avg_nonascii_frac``.  If *ppl_scorer* provided: also ``avg_ppl_adv``,
        ``avg_ppl_clean``, ``ppl_ratio``.
    """
    if len(adversarial_passages) != len(clean_passages):
        raise ValueError("adversarial_passages and clean_passages must be the same length.")

    vdr_vals, zw_fracs, na_fracs = [], [], []
    ppl_adv_vals, ppl_clean_vals = [], []

    for adv, clean in zip(adversarial_passages, clean_passages):
        vdr_vals.append(visual_diff_rate(clean, adv))
        stats = char_class_entropy(adv)
        zw_fracs.append(stats["zerowidth_frac"])
        na_fracs.append(stats["nonascii_frac"])
        if ppl_scorer is not None:
            ppl_adv_vals.append(ppl_scorer.score(adv))   # type: ignore[attr-defined]
            ppl_clean_vals.append(ppl_scorer.score(clean))

    result: dict[str, float] = {
        "visual_diff_rate": sum(vdr_vals) / len(vdr_vals),
        "avg_zerowidth_frac": sum(zw_fracs) / len(zw_fracs),
        "avg_nonascii_frac": sum(na_fracs) / len(na_fracs),
    }
    if ppl_scorer is not None and ppl_adv_vals:
        avg_adv = sum(ppl_adv_vals) / len(ppl_adv_vals)
        avg_clean = sum(ppl_clean_vals) / len(ppl_clean_vals)
        result["avg_ppl_adv"] = avg_adv
        result["avg_ppl_clean"] = avg_clean
        result["ppl_ratio"] = avg_adv / avg_clean if avg_clean > 0 else float("inf")
    return result


def visual_diff_rate(clean: str, adversarial: str) -> float:
    """Fraction of character positions that visibly differ between strings.

    Strips invisible characters from both strings before comparing, so that
    zero-width insertions contribute 0 to the rate.  Uses
    ``difflib.SequenceMatcher`` for an edit-based similarity ratio.

    Args:
        clean: Original passage text.
        adversarial: Perturbed passage text.

    Returns:
        Float in [0, 1]; 0.0 means visually identical (only invisible chars differ).
    """
    vis_clean = _to_visible(clean)
    vis_adv = _to_visible(adversarial)
    if vis_clean == vis_adv:
        return 0.0
    # SequenceMatcher ratio = 2 * M / T where M = matching chars, T = total chars
    ratio = difflib.SequenceMatcher(None, vis_clean, vis_adv).ratio()
    return 1.0 - ratio


def char_class_entropy(text: str) -> dict[str, float]:
    """Compute fractions of ASCII, non-ASCII, zero-width, and control chars.

    Args:
        text: Input string.

    Returns:
        Dict with keys ``ascii_frac``, ``nonascii_frac``, ``zerowidth_frac``,
        ``control_frac``.  All values in [0, 1].  Returns all-zero dict for
        empty strings.
    """
    if not text:
        return {
            "ascii_frac": 0.0,
            "nonascii_frac": 0.0,
            "zerowidth_frac": 0.0,
            "control_frac": 0.0,
        }
    n = len(text)
    ascii_count = 0
    nonascii_count = 0
    zerowidth_count = 0
    control_count = 0
    for ch in text:
        cp = ord(ch)
        if ch in _ALL_INVISIBLE:
            zerowidth_count += 1
        elif cp < 128:
            ascii_count += 1
        elif unicodedata.category(ch).startswith("C"):  # control / format
            control_count += 1
        else:
            nonascii_count += 1
    return {
        "ascii_frac": ascii_count / n,
        "nonascii_frac": nonascii_count / n,
        "zerowidth_frac": zerowidth_count / n,
        "control_frac": control_count / n,
    }


def _to_visible(text: str) -> str:
    """Return *text* with all invisible/non-printable characters stripped.

    Args:
        text: Input string.

    Returns:
        String containing only visible characters.
    """
    return "".join(
        ch for ch in text
        if ch not in _ALL_INVISIBLE and ch.isprintable()
    )
