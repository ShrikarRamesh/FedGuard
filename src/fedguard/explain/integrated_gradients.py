"""Integrated Gradients (Captum) w.r.t. the input window, aggregated per clinical variable.

Channels of the same variable (value + mask + delta) are summed, so the attribution of e.g. "Lactate"
includes the evidence that lactate was (not) measured and how long ago. Baseline = all-zero window, i.e.
every value at its training mean, nothing measured recently, zero statics. Attributions are w.r.t. the
logit (additive completeness holds for the logit, not the sigmoid).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fedguard.data.features import ChannelSpec


def group_index(spec: ChannelSpec) -> tuple[list[str], np.ndarray]:
    """Unique variable names (order of first appearance) and a channel -> group index map."""
    names = list(dict.fromkeys(spec.groups))
    return names, np.array([names.index(g) for g in spec.groups])


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    pad_mask: torch.Tensor,
    n_steps: int = 50,
    internal_batch_size: int = 512,
) -> torch.Tensor:
    """Per-channel attributions [B, L, C] of the logit (model in eval mode)."""
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(lambda inp, m: model(inp, m))
    return ig.attribute(x, baselines=torch.zeros_like(x), additional_forward_args=(pad_mask,), n_steps=n_steps,
                        internal_batch_size=internal_batch_size).detach()  # fmt: skip


def per_variable(attr: np.ndarray, spec: ChannelSpec) -> tuple[list[str], np.ndarray]:
    """[B, L, C] -> ([G names], [B, L, G]) by summing channels of the same variable."""
    names, gidx = group_index(spec)
    out = np.zeros((*attr.shape[:2], len(names)), dtype=np.float32)
    for g in range(len(names)):
        out[..., g] = attr[..., gidx == g].sum(-1)
    return names, out
