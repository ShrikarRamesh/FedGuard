"""Baselines: hand-crafted-feature logistic regression and LightGBM (sanity floor), and a small GRU.

Hand-crafted features are computed from the same causal windows the deep models see (so they inherit the
causality guarantee): per dynamic variable the last (standardised) value, and mean/min/max/slope over the
last 6 h and 24 h of *real* hours; the current measurement mask and the count of measurements in 24 h;
plus the static channels at hour t.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fedguard.data.features import ChannelSpec


def handcrafted_features(x: np.ndarray, pad_mask: np.ndarray, spec: ChannelSpec) -> np.ndarray:
    """x [B, L, C], pad_mask [B, L] -> features [B, F] (float32)."""
    n = spec.n_dyn
    vals = x[:, :, :n]
    has_mask = any(nm.endswith("_mask") for nm in spec.names)
    masks = x[:, :, n : 2 * n] if has_mask else None
    static = x[:, -1, len(spec.names) - _n_static(spec) :]
    feats = [vals[:, -1]]
    L = x.shape[1]
    for w in (6, L):
        w = min(w, L)
        v = vals[:, -w:]
        m = pad_mask[:, -w:, None].astype(np.float32)
        cnt = np.maximum(m.sum(1), 1.0)
        mean = (v * m).sum(1) / cnt
        big = np.where(m > 0, v, np.inf).min(1)
        small = np.where(m > 0, v, -np.inf).max(1)
        feats += [mean, np.where(np.isfinite(big), big, 0.0), np.where(np.isfinite(small), small, 0.0)]
        # least-squares slope over real hours in the window (0 if < 2 real hours)
        tt = np.arange(w, dtype=np.float32)[None, :, None]
        tm = (tt * m).sum(1) / cnt
        cov = (((tt - tm[:, None]) * (v - mean[:, None])) * m).sum(1)
        var = (((tt - tm[:, None]) ** 2) * m).sum(1)
        feats.append(np.where(var > 0, cov / np.maximum(var, 1e-6), 0.0))
    if masks is not None:
        feats += [masks[:, -1], masks.sum(1)]
    feats.append(static)
    return np.concatenate(feats, axis=1).astype(np.float32)


def _n_static(spec: ChannelSpec) -> int:
    """Static channels are everything after the value block that is not a mask or delta channel."""
    return sum(1 for nm in spec.names[spec.n_dyn :] if not nm.endswith(("_mask", "_delta")))


class GRUClassifier(nn.Module):
    """Small GRU over the window; the last real hour's hidden state feeds a 2-layer head."""

    def __init__(self, n_channels: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)  # fmt: skip
        self.drop = nn.Dropout(dropout)
        self.head1 = nn.Linear(hidden, hidden)
        self.head2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # Windows are left-padded, so the last timestep is always the current (real) hour. Padded steps
        # feed zeros; through the biases they still move the hidden state, so the GRU can tell stay length
        # apart (ICULOS is also an input). Acceptable for a baseline; PatchTST masks padding exactly.
        out, _ = self.gru(x * pad_mask[..., None].to(x.dtype))
        return self.head2(self.drop(torch.relu(self.head1(out[:, -1])))).squeeze(-1)
