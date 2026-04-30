"""Global random seed management.

Call ``set_seed(42)`` at the top of every experiment script to ensure
reproducible results across Python random, NumPy, and PyTorch.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.  Default 42 matches PoisonedRAG.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
