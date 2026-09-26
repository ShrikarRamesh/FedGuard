"""Attention rollout (Abnar & Zuidema, 2020) for the custom PatchTST, mapped back to hours.

Per layer the attention is averaged over heads, the residual connection is modelled as 0.5 A + 0.5 I
(rows renormalised), and layers are multiplied. The CLS row of the product gives each patch's share of
the summary token; an hour's importance is the sum over patches containing it, divided by patch_len.
"""

from __future__ import annotations

import numpy as np
import torch

from fedguard.models.patchtst import PatchTST


@torch.no_grad()
def rollout(model: PatchTST, x: torch.Tensor, pad_mask: torch.Tensor) -> np.ndarray:
    """Hour importances [B, L] (rows sum to ~1 over real hours)."""
    model.eval()
    model.set_keep_attention(True)
    try:
        model(x, pad_mask)
        maps = model.attention_maps()
    finally:
        model.set_keep_attention(False)
    b, t = maps[0].shape[0], maps[0].shape[-1]
    eye = torch.eye(t, device=x.device).expand(b, t, t)
    joint = eye.clone()
    for a in maps:
        a = a.mean(1)  # heads
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        joint = a @ joint
    cls_to_patch = joint[:, 0, 1:]  # [B, P]
    L, pl, st = model.lookback, model.patch_len, model.stride
    hours = torch.zeros(b, L, device=x.device)
    for p in range(model.n_patches):
        hours[:, p * st : p * st + pl] += cls_to_patch[:, p : p + 1] / pl
    hours = hours * pad_mask
    return (hours / hours.sum(-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
