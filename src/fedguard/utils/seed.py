"""Deterministic seeding for Python, NumPy and torch."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA).

    With ``deterministic_torch`` cuDNN autotuning is disabled and deterministic kernels are
    requested where available (``warn_only`` so unsupported ops do not crash). Remaining
    nondeterminism (e.g. some CUDA scatter/index_add kernels) is documented in PROGRESS.md.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def derived_seed(seed: int, *keys: object) -> int:
    """Stable 32-bit seed derived from a base seed and arbitrary keys (order-independent of process state).

    Uses SHA-256 rather than ``hash()`` so it is identical across processes and platforms.
    """
    h = hashlib.sha256(repr((seed, *keys)).encode()).digest()
    return int.from_bytes(h[:4], "little")


def rng_for(seed: int, *keys: object) -> np.random.Generator:
    """NumPy Generator seeded from ``derived_seed(seed, *keys)``."""
    return np.random.default_rng(derived_seed(seed, *keys))
