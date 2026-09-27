"""Training / inference loops shared by centralized, local and federated training.

* Loss: BCE-with-logits with positive-class weighting (n_neg / n_pos of the *training* labels). A
  weighted sampler is avoided because it is incompatible with Opacus Poisson sampling. Optional focal loss.
* AdamW, gradient-norm clipping for non-DP runs, early stopping on validation AUPRC.
* fp32 throughout (AMP interacts badly with Opacus).
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from fedguard.data.windows import WindowDataset
from fedguard.eval import metrics as M
from fedguard.models.baselines import GRUClassifier
from fedguard.models.patchtst import build_patchtst
from fedguard.models.uncertainty import mc_predict


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def collate_windows(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate either a pre-batched ``(x, m, y)`` from ``__getitems__`` or a list of single samples."""
    if (
        isinstance(batch, tuple)
        and len(batch) == 3
        and isinstance(batch[0], np.ndarray)
        and batch[0].ndim == 3
    ):
        x, m, y = batch
    else:
        x = np.stack([b[0] for b in batch])
        m = np.stack([b[1] for b in batch])
        y = np.asarray([b[2] for b in batch], dtype=np.float32)
    return (
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.asarray(m)),
        torch.from_numpy(np.asarray(y)),
    )


def make_loader(ds, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0) -> DataLoader:
    """DataLoader that fetches whole batches via ``__getitems__`` (vectorised gather)."""
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(ds, generator=g) if shuffle else SequentialSampler(ds)
    return DataLoader(
        ds, batch_sampler=BatchSampler(sampler, batch_size, drop_last=False), collate_fn=collate_windows,
        num_workers=num_workers,
    )  # fmt: skip


def build_model(model_cfg, n_channels: int, lookback: int) -> nn.Module:
    """Model factory from the ``model`` config section."""
    if model_cfg.name == "patchtst":
        return build_patchtst(model_cfg, n_channels, lookback)
    if model_cfg.name == "gru":
        return GRUClassifier(n_channels, hidden=model_cfg.get("hidden", 64), dropout=model_cfg.dropout)
    raise ValueError(f"unknown deep model {model_cfg.name!r} (lr/lgbm are trained by train.sklearn_models)")


def pos_weight_from(labels: np.ndarray) -> float:
    """n_neg / n_pos of the training labels (1.0 if there are no positives)."""
    n_pos = float(np.sum(labels))
    return float((len(labels) - n_pos) / n_pos) if n_pos > 0 else 1.0


def make_loss(
    kind: str, pos_weight: float, gamma: float = 2.0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    pw = torch.tensor(pos_weight)

    def bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw.to(logits.device))

    def focal(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pw.to(logits.device))
        p = torch.sigmoid(logits)
        pt = torch.where(y > 0.5, p, 1 - p)
        return ((1 - pt) ** gamma * ce).mean()

    if kind == "bce":
        return bce
    if kind == "focal":
        return focal
    raise ValueError(f"unknown loss {kind!r}")


@dataclass
class ProxTerm:
    """FedProx proximal term (mu / 2) * ||w - w_global||^2 (Li et al., 2020)."""

    ref: list[torch.Tensor]
    mu: float

    def __call__(self, model: nn.Module) -> torch.Tensor:
        return (
            0.5 * self.mu * sum(((p - r) ** 2).sum() for p, r in zip(trainable(model), self.ref, strict=True))
        )


def trainable(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def train_epoch(
    model: nn.Module, loader, opt: torch.optim.Optimizer, loss_fn, device: torch.device,
    grad_clip: float | None = None, prox: ProxTerm | None = None, max_steps: int | None = None,
) -> dict[str, float]:  # fmt: skip
    """One pass over ``loader`` (or ``max_steps`` batches). Returns mean loss and number of steps/samples."""
    model.train()
    tot, steps, n = 0.0, 0, 0
    for x, m, y in loader:
        if len(y) == 0:  # Opacus Poisson sampling can produce empty batches
            continue
        x, m, y = (
            x.to(device, non_blocking=True),
            m.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, m), y)
        if prox is not None:
            loss = loss + prox(model)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        tot += float(loss.detach()) * len(y)
        n += len(y)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {"loss": tot / max(n, 1), "steps": steps, "samples": n}


# Set by the run drivers to <run dir>/heartbeat; long inference loops touch it so scripts/watchdog.py can tell a
# slow phase (e.g. MC-Dropout over val + test) from a hung job.
HEARTBEAT: Path | None = None


def _beat(i: int, every: int = 20) -> None:
    if HEARTBEAT is not None and i % every == 0:
        HEARTBEAT.touch()


@torch.no_grad()
def predict(model: nn.Module, ds: WindowDataset, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    """Deterministic probabilities (eval mode) for every sample of ``ds`` in order."""
    model.eval()
    out = []
    for i, (x, m, _) in enumerate(make_loader(ds, batch_size, shuffle=False)):
        _beat(i)
        out.append(torch.sigmoid(model(x.to(device), m.to(device))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@torch.no_grad()
def predict_mc(
    model: nn.Module, ds: WindowDataset, device: torch.device, T: int, batch_size: int = 512, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """MC-Dropout mean and std for every sample of ``ds`` in order."""
    torch.manual_seed(seed)
    means, stds = [], []
    for i, (x, m, _) in enumerate(make_loader(ds, batch_size, shuffle=False)):
        _beat(i)
        mu, sd = mc_predict(model, x.to(device), m.to(device), T=T)
        means.append(mu.float().cpu().numpy())
        stds.append(sd.float().cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)


def fit(
    model: nn.Module, train_ds: WindowDataset, val_ds: WindowDataset, tcfg, device: torch.device, seed: int,
    ckpt_dir: Path | None = None, log: Callable[[dict], None] | None = None, eval_batch: int = 2048,
) -> dict[str, Any]:  # fmt: skip
    """Train with early stopping on validation AUPRC; returns history and restores the best weights."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    pw = pos_weight_from(train_ds.labels()) if tcfg.pos_weight == "auto" else float(tcfg.pos_weight)
    loss_fn = make_loss(tcfg.loss, pw, tcfg.get("focal_gamma", 2.0))
    best, best_state, bad, history = -np.inf, copy.deepcopy(model.state_dict()), 0, []
    y_val = val_ds.labels().astype(np.float64)
    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()
        loader = make_loader(train_ds, tcfg.batch_size, shuffle=True, seed=seed * 1000 + epoch)
        tr = train_epoch(model, loader, opt, loss_fn, device, grad_clip=tcfg.grad_clip,
                         max_steps=tcfg.get("max_steps_per_epoch"))  # fmt: skip
        p_val = predict(model, val_ds, device, eval_batch)
        val = M.summary(y_val, p_val)
        score = val["auprc"] if np.isfinite(val["auprc"]) else -val.get("brier", 0.0)
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_auroc": val["auroc"], "val_auprc": val["auprc"],
               "val_brier": val["brier"], "epoch_s": time.time() - t0}  # fmt: skip
        history.append(row)
        if log:
            log(row)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            if ckpt_dir is not None:
                torch.save({"model": best_state, "epoch": epoch, "val": val}, ckpt_dir / "best.pt")
        else:
            bad += 1
        if ckpt_dir is not None:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
                       ckpt_dir / "last.pt")  # fmt: skip
        if bad >= tcfg.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return {"history": history, "best_val_auprc": best, "pos_weight": pw, "epochs_run": len(history)}
