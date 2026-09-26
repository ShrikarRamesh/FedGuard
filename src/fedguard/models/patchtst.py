"""Own PatchTST variant for irregular ICU time series, fully Opacus-compatible.

Differences from the original PatchTST (Nie et al., ICLR 2023), all deliberate (docs/decisions.md):
* **Channel-mixing patches:** each patch (patch_len hours x C channels) is flattened and projected with
  one ``nn.Linear``. Channel-independent PatchTST would run ~107 separate sequences per sample.
* **LayerNorm, not BatchNorm** (BatchNorm mixes samples, which breaks per-sample DP gradients).
* **Hand-written attention** from ``nn.Linear`` Q/K/V/out, so Opacus computes per-sample gradients for
  every parameter and attention weights are available for rollout explanations.
* Positional and CLS embeddings are ``nn.Embedding`` indexed with **batch-expanded** id tensors (D1):
  indexing with a bare ``arange`` breaks Opacus' per-sample gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


def num_patches(lookback: int, patch_len: int, stride: int) -> int:
    if lookback < patch_len:
        raise ValueError(f"lookback {lookback} < patch_len {patch_len}")
    if (lookback - patch_len) % stride:
        raise ValueError(
            f"(lookback - patch_len) must be divisible by stride: {lookback}, {patch_len}, {stride}"
        )
    return (lookback - patch_len) // stride + 1


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built from nn.Linear; keeps the last attention map for explanations."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.keep_attn = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(b, t, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)  # [B, H, T, T]
        scores = scores.masked_fill(~key_valid[:, None, None, :], NEG_INF)
        attn = scores.softmax(dim=-1)
        if self.keep_attn:
            self.last_attn = attn.detach()
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, t, d)
        return self.o(out)


class EncoderBlock(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), key_valid))
        return x + self.drop3(self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x))))))


class PatchTST(nn.Module):
    """x [B, L, C] + pad_mask [B, L] (True = real hour) -> one logit per sample."""

    def __init__(
        self, n_channels: int, lookback: int = 24, patch_len: int = 4, patch_stride: int = 2,
        d_model: int = 128, n_layers: int = 4, n_heads: int = 8, d_ff: int = 256, dropout: float = 0.2,
        head_hidden: int = 64,
    ):  # fmt: skip
        super().__init__()
        self.lookback, self.patch_len, self.stride = lookback, patch_len, patch_stride
        self.n_patches = num_patches(lookback, patch_len, patch_stride)
        self.embed = nn.Linear(patch_len * n_channels, d_model)
        self.pos = nn.Embedding(self.n_patches, d_model)
        self.cls = nn.Embedding(1, d_model)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head1 = nn.Linear(d_model, head_hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head2 = nn.Linear(head_hidden, 1)

    def patchify(self, x: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, L, C] -> patches [B, P, patch_len*C] and patch validity [B, P] (any real hour in the patch)."""
        b, l_, c = x.shape
        if l_ != self.lookback:
            raise ValueError(f"expected lookback {self.lookback}, got {l_}")
        p = x.unfold(1, self.patch_len, self.stride)  # [B, P, C, patch_len]
        p = p.permute(0, 1, 3, 2).reshape(b, self.n_patches, self.patch_len * c)
        valid = pad_mask.unfold(1, self.patch_len, self.stride).any(-1)  # [B, P]
        return p, valid

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x * pad_mask.unsqueeze(-1).to(x.dtype)  # padded hours can never influence the output
        patches, valid = self.patchify(x, pad_mask)
        pos_ids = torch.arange(self.n_patches, device=x.device).unsqueeze(0).expand(b, -1)  # D1
        cls_ids = torch.zeros(b, 1, dtype=torch.long, device=x.device)
        h = torch.cat([self.cls(cls_ids), self.embed(patches) + self.pos(pos_ids)], dim=1)
        key_valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), valid], dim=1)
        h = self.in_drop(h)
        for blk in self.blocks:
            h = blk(h, key_valid)
        z = self.ln_f(h[:, 0])
        return self.head2(self.head_drop(F.gelu(self.head1(z)))).squeeze(-1)

    def set_keep_attention(self, keep: bool) -> None:
        for blk in self.blocks:
            blk.attn.keep_attn = keep
            blk.attn.last_attn = None

    def attention_maps(self) -> list[torch.Tensor]:
        """Last forward's attention per layer, each [B, H, 1+P, 1+P] (requires set_keep_attention(True))."""
        return [blk.attn.last_attn for blk in self.blocks]  # type: ignore[misc]


def build_patchtst(model_cfg, n_channels: int, lookback: int) -> PatchTST:
    """Construct from a ``model`` config section."""
    return PatchTST(
        n_channels=n_channels, lookback=lookback, patch_len=model_cfg.patch_len, patch_stride=model_cfg.patch_stride,
        d_model=model_cfg.d_model, n_layers=model_cfg.n_layers, n_heads=model_cfg.n_heads, d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout, head_hidden=model_cfg.head_hidden,
    )  # fmt: skip
