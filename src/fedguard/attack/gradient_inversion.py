"""Gradient-inversion attack (DLG / iDLG style, cosine gradient matching) against one client update.

Threat model (attacker's best case, D23): an honest-but-curious server observes the gradient of the loss
for a SINGLE patient window (batch size 1) at the current global model, computed without dropout; it knows
the architecture, loss (incl. pos_weight) and the padding mask (full 24-hour window).

Conditions: (a) raw gradient (no DP); (b) clipped to C only; (c) DP-SGD release clip(g, C) + N(0, sigma^2 C^2 I)
with sigma = the noise multiplier FedGuard actually uses for a client at total epsilon in {1, 3, 8}.

Attack: label inferred from the sign of the output-bias gradient (iDLG); dummy input optimised with Adam to
minimise 1 - cos(grad(dummy), observed) (Geiping et al., 2020), several random restarts, best by final loss.
Metrics on the value channels of the vitals: per-variable Pearson r over the 24 hours (variables whose
true series is not constant) and MSE (standardised units).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from fedguard.data.features import DYN, VITALS
from fedguard.train.loops import make_loss

VITAL_IDX = [DYN.index(v) for v in VITALS if v != "EtCO2"]  # EtCO2 is almost never measured


def flat_grad(model: nn.Module, loss: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([x.reshape(-1) for x in g])


def observed_gradient(model: nn.Module, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor, loss_fn,
                      clip: float | None, sigma: float, gen: torch.Generator) -> torch.Tensor:  # fmt: skip
    """What the server sees for one window: raw, clipped, or clipped + Gaussian noise (DP-SGD, one sample)."""
    g = flat_grad(model, loss_fn(model(x, m), y)).detach()
    if clip is not None:
        g = g * min(1.0, clip / (float(g.norm()) + 1e-12))
    if sigma > 0:
        g = g + torch.randn(g.shape, generator=gen, device="cpu").to(g.device) * sigma * (clip or 1.0)
    return g


def infer_label(model: nn.Module, g_obs: torch.Tensor) -> float:
    """iDLG for a single-logit BCE head: d loss / d bias_out = w * (sigmoid(z) - y) is < 0 iff y = 1."""
    return 1.0 if float(g_obs[-1]) < 0 else 0.0  # the output bias is the last parameter


@dataclass
class AttackResult:
    x_rec: np.ndarray
    loss: float
    label_true: float
    label_inferred: float


def invert(model: nn.Module, g_obs: torch.Tensor, m: torch.Tensor, shape: tuple[int, ...], loss_fn,
           iters: int = 400, restarts: int = 3, lr: float = 0.1, seed: int = 0, y_true: float = 0.0) -> AttackResult:  # fmt: skip
    y_hat = infer_label(model, g_obs)
    y_t = torch.tensor([y_hat], device=g_obs.device)
    best: tuple[float, torch.Tensor] = (np.inf, torch.zeros(shape))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(restarts):
        dummy = torch.randn(shape, generator=gen).to(g_obs.device).requires_grad_(True)
        opt = torch.optim.Adam([dummy], lr=lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * 0.6), int(iters * 0.85)], gamma=0.1)
        for _ in range(iters):
            opt.zero_grad()
            g = flat_grad(model, loss_fn(model(dummy, m), y_t), create_graph=True)
            loss = 1 - nn.functional.cosine_similarity(g, g_obs, dim=0)
            loss.backward()
            opt.step()
            sched.step()
        final = float(loss.detach())
        if final < best[0]:
            best = (final, dummy.detach().cpu())
    return AttackResult(best[1].numpy()[0], best[0], y_true, y_hat)


def reconstruction_metrics(x_true: np.ndarray, x_rec: np.ndarray) -> dict[str, float]:
    """Pearson r per vital (non-constant true series) and MSE over the vital value channels."""
    rs = {}
    for i in VITAL_IDX:
        a, b = x_true[:, i], x_rec[:, i]
        if a.std() > 1e-6 and b.std() > 1e-6:
            rs[DYN[i]] = float(np.corrcoef(a, b)[0, 1])
        elif a.std() > 1e-6:
            rs[DYN[i]] = 0.0
    mse = float(np.mean((x_true[:, VITAL_IDX] - x_rec[:, VITAL_IDX]) ** 2))
    return {
        "mean_r": float(np.mean(list(rs.values()))) if rs else float("nan"),
        "mse": mse,
        **{f"r_{k}": v for k, v in rs.items()},
    }


def attack_condition(model: nn.Module, windows: list[tuple[np.ndarray, np.ndarray, float]], pos_weight: float,
                     clip: float | None, sigma: float, device: torch.device, iters: int, restarts: int,
                     seed: int) -> list[dict]:  # fmt: skip
    """Run the attack on each (x, mask, y) window under one condition."""
    loss_fn = make_loss("bce", pos_weight)
    model.eval()
    out = []
    for k, (x, m, y) in enumerate(windows):
        xt = torch.from_numpy(x[None]).to(device)
        mt = torch.from_numpy(m[None]).to(device)
        yt = torch.tensor([y], device=device)
        gen = torch.Generator().manual_seed(seed * 1000 + k)
        g_obs = observed_gradient(model, xt, mt, yt, loss_fn, clip, sigma, gen)
        res = invert(model, g_obs, mt, tuple(xt.shape), loss_fn, iters=iters, restarts=restarts, seed=seed * 1000 + k,
                     y_true=float(y))  # fmt: skip
        met = reconstruction_metrics(x, res.x_rec)
        out.append({"k": k, **met, "label_true": float(y), "label_inferred": res.label_inferred,
                    "match_loss": res.loss, "x_rec_vitals": res.x_rec[:, VITAL_IDX].tolist()})  # fmt: skip
    return out
