"""One federated client: holds only its own data, trains locally (optionally with patient-level DP-SGD).

Non-DP local work: ``local_steps`` AdamW steps on shuffled windows of the client's own training set
(D16), optional FedProx proximal term. Optimizer state is reset every participation (standard FedAvg).

DP local work (D2, D3, D17): the dataset element is a *patient*. Opacus Poisson-samples patients at rate
q = B / n_patients; each sampled patient contributes one window at a uniformly random hour of its own
stay; per-sample gradients are clipped to C and Gaussian noise is added. The noise multiplier is
calibrated so that ``local_epochs * r_max`` patient-epochs spend the client's target epsilon; the RDP
accountant persists across participations and reports the *total* epsilon spent. AdamW is applied to
the privatised gradients (post-processing, so the guarantee is unchanged).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fedguard.data.windows import ClientArrays, WindowDataset
from fedguard.train import loops
from fedguard.utils.seed import derived_seed


class PatientWindowDataset:
    """Dataset over patients; item i = one window at a uniformly random hour of patient i's stay.

    The random hour depends only on patient i's own record and an independent RNG, so each sampled patient
    contributes exactly one (clipped) gradient per step: add/remove-one-patient sensitivity is C.
    """

    def __init__(self, arrays: ClientArrays, lookback: int, seed: int):
        self.windows = WindowDataset(arrays, lookback)
        self.starts = arrays.offsets[:-1]
        self.lens = np.diff(arrays.offsets)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.lens)

    def _rows(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return self.starts[idx] + (self.rng.random(len(idx)) * self.lens[idx]).astype(np.int64)

    def __getitem__(self, i: int):
        x, m, y = self.windows.gather(self._rows(np.array([i])))
        return x[0], m[0], y[0]

    def __getitems__(self, indices: list[int]):
        return self.windows.gather(self._rows(np.asarray(indices, dtype=np.int64)))


@dataclass
class DPState:
    target_eps: float | None  # None for the equal-noise rule (eps follows from the shared noise)
    delta: float
    noise_multiplier: float
    sample_rate: float
    planned_steps: int
    r_max: int
    local_epochs: int
    physical_batch: int | None
    engine: Any
    optimizer: Any
    loader: Any
    participations: int = 0
    exhausted: bool = False


@dataclass
class FLClient:
    name: str
    train_arrays: ClientArrays
    lookback: int
    model: nn.Module
    device: torch.device
    seed: int
    lr: float
    weight_decay: float
    batch_size: int
    local_steps: int
    grad_clip: float | None
    pos_weight: float
    prox_mu: float = 0.0
    dp: DPState | None = None
    _round: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        """Training windows (FedAvg weight n_i)."""
        return int(self.train_arrays.offsets[-1])

    @property
    def n_patients(self) -> int:
        return int(self.train_arrays.n_patients)

    def core(self) -> nn.Module:
        """Underlying nn.Module (unwrapped from Opacus' GradSampleModule if present)."""
        return getattr(self.model, "_module", self.model)

    def epsilon(self) -> float:
        """Total epsilon spent so far (0 before the first DP step, 0 for non-DP clients)."""
        if self.dp is None or self.dp.participations == 0:
            return 0.0
        return float(self.dp.engine.get_epsilon(self.dp.delta))

    def can_participate(self) -> bool:
        return self.dp is None or not self.dp.exhausted

    def local_train(self, global_state: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Load the global model, train locally, return the new state and bookkeeping."""
        if not self.can_participate():
            raise RuntimeError(f"client {self.name} has exhausted its privacy budget")
        self._round += 1
        # Seed dropout / DP noise per (client, participation) so local training does not depend on how many
        # other clients ran before it in this process (in-house engine == Flower deployment, D19).
        torch.manual_seed(derived_seed(self.seed, "local", self._round))
        self.core().load_state_dict(global_state)
        loss_fn = loops.make_loss("bce", self.pos_weight)
        stats = self._plain_train(loss_fn) if self.dp is None else self._dp_train(loss_fn)
        stats["eps"] = self.epsilon()
        self.history.append({"round": self._round, **stats})
        state = {k: v.detach().cpu().clone() for k, v in self.core().state_dict().items()}
        return {"state": state, "n": self.n_samples, **stats}

    def _batches(self) -> Iterator:
        ds = WindowDataset(self.train_arrays, self.lookback)
        epoch = 0
        while True:
            epoch += 1
            yield from loops.make_loader(ds, self.batch_size, shuffle=True,
                                         seed=(self.seed * 100003 + self._round) * 1009 + epoch)  # fmt: skip

    def _plain_train(self, loss_fn) -> dict[str, Any]:
        core = self.core()
        opt = torch.optim.AdamW(core.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        prox = None
        if self.prox_mu > 0:
            prox = loops.ProxTerm([p.detach().clone() for p in loops.trainable(core)], self.prox_mu)
        batches = self._batches()
        loader = (next(batches) for _ in range(self.local_steps))
        return loops.train_epoch(core, loader, opt, loss_fn, self.device, grad_clip=self.grad_clip, prox=prox)

    def _dp_train(self, loss_fn) -> dict[str, Any]:
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        dp = self.dp
        assert dp is not None
        loss_sum, steps, samples = 0.0, 0, 0
        self.model.train()
        for _ in range(dp.local_epochs):
            use_bmm = dp.physical_batch is not None
            ctx = (
                BatchMemoryManager(
                    data_loader=dp.loader, max_physical_batch_size=dp.physical_batch, optimizer=dp.optimizer
                )
                if use_bmm
                else contextlib.nullcontext(dp.loader)
            )
            with ctx as loader:
                for x, m, y in loader:
                    x, m, y = x.to(self.device), m.to(self.device), y.to(self.device)
                    dp.optimizer.zero_grad(set_to_none=True)
                    if len(y):
                        loss = loss_fn(self.model(x, m), y)
                        loss.backward()
                        loss_sum += float(loss.detach()) * len(y)
                        samples += len(y)
                    else:  # empty Poisson batch: still a (noise-only) step that the accountant counts
                        _zero_grad_samples(self.model)
                    dp.optimizer.step()
                    if not getattr(dp.optimizer, "_is_last_step_skipped", False):
                        steps += 1
        dp.participations += 1
        eps = self.epsilon()
        if dp.participations >= dp.r_max or (dp.target_eps is not None and eps >= dp.target_eps):
            dp.exhausted = True
        return {"loss": loss_sum / max(samples, 1), "steps": steps, "samples": samples}


def _zero_grad_samples(model: nn.Module) -> None:
    """For an empty Poisson batch give every parameter an empty per-sample gradient so Opacus can add noise."""
    for p in model.parameters():
        if p.requires_grad:
            p.grad_sample = torch.zeros((0, *p.shape), device=p.device, dtype=p.dtype)


def setup_dp(
    client: FLClient, target_eps: float | None, noise_multiplier: float, delta: float, logical_batch: int,
    r_max: int, local_epochs: int, max_grad_norm: float, physical_batch: int | None, lr: float,
) -> None:  # fmt: skip
    """Wrap the client's model/optimizer/loader with Opacus (Poisson sampling over patients, RDP accountant)."""
    from opacus import PrivacyEngine

    from fedguard.privacy.accounting import check_delta, planned_steps

    check_delta(delta, client.n_patients)
    q, steps = planned_steps(client.n_patients, logical_batch, local_epochs, r_max)
    ds = PatientWindowDataset(client.train_arrays, client.lookback, seed=client.seed * 7919 + 17)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=logical_batch, shuffle=False, collate_fn=loops.collate_windows
    )
    opt = torch.optim.AdamW(client.model.parameters(), lr=lr, weight_decay=client.weight_decay)
    engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, opt, loader = engine.make_private(
            module=client.model, optimizer=opt, data_loader=loader, noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm, poisson_sampling=True,
        )  # fmt: skip
    client.model = model
    pb = physical_batch if physical_batch is not None and physical_batch < logical_batch else None
    client.dp = DPState(
        target_eps, delta, noise_multiplier, q, steps, r_max, local_epochs, pb, engine, opt, loader
    )
