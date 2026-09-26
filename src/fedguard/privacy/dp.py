"""Patient-level DP-SGD with k windows per sampled patient (extension of D2; D25).

Opacus computes per-*sample* gradients; it cannot group several windows of one patient. Here the per-patient
gradient is computed directly with ``torch.func`` (vmap over patients of the gradient of the patient's mean
loss over k windows), then:

    g_i  = clip(grad of patient i's mean loss over its k windows, C)       (sensitivity C per patient)
    g    = (sum_{i in Poisson batch} g_i + N(0, sigma^2 C^2 I)) / B        (B = expected batch size q * n)
    theta <- AdamW(theta, g)                                              (post-processing)

Patients are Poisson-sampled with rate q every step, and the k windows are drawn uniformly (with replacement)
from the patient's own stay with an RNG independent of other patients. Each patient therefore contributes
one clipped vector per step, so the subsampled Gaussian RDP analysis of Opacus' ``RDPAccountant`` applies
unchanged; the accountant is stepped with (sigma, q) after every step. k = 1 reproduces standard DP-SGD
(tested against Opacus' per-sample clipping in tests/test_privacy.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from opacus.accountants import RDPAccountant
from torch.func import functional_call, grad, vmap

from fedguard.data.windows import ClientArrays, WindowDataset


def per_patient_clipped_sum(
    model: nn.Module, loss_fn, x: torch.Tensor, m: torch.Tensor, y: torch.Tensor, clip: float
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """x [P, k, L, C], m [P, k, L], y [P, k] -> (sum over patients of clipped per-patient grads, per-patient norms).

    Each patient's gradient is the gradient of its mean loss over its k windows. Dropout draws independent
    masks per patient (vmap randomness="different")."""
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}

    def patient_loss(prm, xk, mk, yk):
        return loss_fn(functional_call(model, (prm, buffers), (xk, mk)), yk)

    g = vmap(grad(patient_loss), in_dims=(None, 0, 0, 0), randomness="different")(params, x, m, y)
    norms = torch.sqrt(sum((t.reshape(t.shape[0], -1) ** 2).sum(1) for t in g.values()))
    factor = (clip / (norms + 1e-6)).clamp(max=1.0)
    summed = {n: torch.einsum("p,p...->...", factor, t) for n, t in g.items()}
    return summed, norms


@dataclass
class PatientDPSGD:
    """One client's patient-level DP-SGD state (optimizer, accountant, sampling RNG)."""

    model: nn.Module
    arrays: ClientArrays
    lookback: int
    noise_multiplier: float
    sample_rate: float
    max_grad_norm: float
    windows_per_patient: int
    lr: float
    weight_decay: float
    seed: int
    device: torch.device
    accountant: RDPAccountant = field(default_factory=RDPAccountant)
    steps: int = 0

    def __post_init__(self) -> None:
        self.ds = WindowDataset(self.arrays, self.lookback)
        self.starts = self.arrays.offsets[:-1]
        self.lens = np.diff(self.arrays.offsets)
        self.n = len(self.lens)
        self.rng = np.random.default_rng(self.seed)
        self.noise_gen = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr,
                                     weight_decay=self.weight_decay)  # fmt: skip

    @property
    def expected_batch(self) -> float:
        return self.sample_rate * self.n

    def epsilon(self, delta: float) -> float:
        return float(self.accountant.get_epsilon(delta)) if self.steps else 0.0

    def step(self, loss_fn, chunk: int = 256) -> dict[str, float]:
        """One Poisson-sampled DP-SGD step. Returns the number of sampled patients and mean clip factor."""
        k = self.windows_per_patient
        sampled = np.flatnonzero(self.rng.random(self.n) < self.sample_rate)
        params = [p for p in self.model.parameters() if p.requires_grad]
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        total = {n: torch.zeros_like(p) for n, p in zip(names, params, strict=True)}
        clipped_frac, loss_sum = [], 0.0
        self.model.train()
        for s in range(0, len(sampled), chunk):
            pats = sampled[s : s + chunk]
            rows = self.starts[pats][:, None] + (
                self.rng.random((len(pats), k)) * self.lens[pats][:, None]
            ).astype(np.int64)
            x, m, y = self.ds.gather(rows.reshape(-1))
            L, C = x.shape[1], x.shape[2]
            xt = torch.from_numpy(x).to(self.device).view(len(pats), k, L, C)
            mt = torch.from_numpy(m).to(self.device).view(len(pats), k, L)
            yt = torch.from_numpy(y).to(self.device).view(len(pats), k)
            summed, norms = per_patient_clipped_sum(self.model, loss_fn, xt, mt, yt, self.max_grad_norm)
            for n in total:
                total[n] += summed[n]
            clipped_frac.append((norms > self.max_grad_norm).float().mean().item())
            with torch.no_grad():
                loss_sum += float(loss_fn(self.model(xt.view(-1, L, C), mt.view(-1, L)), yt.view(-1))) * len(
                    pats
                )
        sd = self.noise_multiplier * self.max_grad_norm
        self.opt.zero_grad(set_to_none=True)
        for n, p in zip(names, params, strict=True):
            noise = torch.normal(0.0, sd, size=p.shape, device=self.device, generator=self.noise_gen)
            p.grad = (total[n] + noise) / self.expected_batch
        self.opt.step()
        self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)
        self.steps += 1
        return {"patients": float(len(sampled)), "clipped_frac": float(np.mean(clipped_frac)) if clipped_frac else 0.0,
                "loss": loss_sum / max(len(sampled), 1)}  # fmt: skip
