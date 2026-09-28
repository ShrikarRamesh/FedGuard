"""In-house event-driven FL simulator with a simulated clock (sync FedAvg/FedProx and async FedGuard).

Real computation (local training) happens for real; only *time* is simulated, so sync vs async comparisons
are fair and deterministic for a fixed seed:

    job duration  = speed_i * (samples processed / 1000) * time_per_1k * max(0.5, 1 + jitter * N(0,1))
                    + (model bytes down + up) / bandwidth

Sync (FedAvg / FedProx): each round the server sends the global model to every online client that still
has budget and waits for all of them. A client that goes offline during its job never reports; the server
then waits until ``sync_timeout`` (from round start) before aggregating what arrived (D18). Aggregation is
weighted by n_i (training windows).

Async (FedGuard): the server merges each update on arrival with a_i = alpha0 * (n_i/N) * s(tau) and the
client immediately starts again from the newest global model. Updates staler than ``max_staleness`` are
dropped. A client offline mid-job loses that update and restarts when it comes back.

Every event is appended to ``events.jsonl`` (replayed by the Streamlit "Train together" page).
"""

from __future__ import annotations

import copy
import heapq
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from fedguard.data.scenario import NormMode, Scenario
from fedguard.data.windows import WindowDataset
from fedguard.eval import metrics as M
from fedguard.fl import aggregators as agg
from fedguard.fl.client import FLClient, setup_dp
from fedguard.train import loops
from fedguard.utils.io import append_jsonl


@dataclass
class EngineResult:
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    best_val_auprc: float
    evals: list[dict[str, Any]]
    sim_time: float
    versions: int
    bytes_total: int
    client_summary: dict[str, dict[str, Any]]
    budgets: dict[str, Any] = field(default_factory=dict)


class FLEngine:
    def __init__(self, sc: Scenario, cfg: DictConfig, run_dir: Path, seed: int, device: torch.device):
        self.sc, self.cfg, self.fl, self.seed, self.device = sc, cfg, cfg.fl, seed, device
        self.events_path = Path(run_dir) / "events.jsonl"
        self.events_path.unlink(missing_ok=True)
        self.dp_cfg = cfg.privacy
        self.norm: NormMode = "public" if (self.dp_cfg.enabled or self.fl.get("norm") == "public") else "site"
        self.rng = np.random.default_rng(seed * 7 + 3)
        self.t0_wall = time.time()
        torch.manual_seed(seed)
        names = sc.client_names
        self.global_model = loops.build_model(cfg.model, self._spec_channels(), sc.cfg.lookback).to(device)
        self.global_state = _cpu_state(self.global_model)
        self.model_bytes = agg.state_bytes(self.global_state)
        speeds = list(self.fl.client_speed)
        self.speed = {c: float(speeds[i % len(speeds)]) for i, c in enumerate(names)}
        self.clients: dict[str, FLClient] = {}
        pos_weight_cfg = self.dp_cfg.get("pos_weight", 10.0) if self.dp_cfg.enabled else None
        for i, c in enumerate(names):
            arr = sc.arrays(c, "train", self.norm)
            pw = float(pos_weight_cfg) if pos_weight_cfg is not None else loops.pos_weight_from(arr.label)
            self.clients[c] = FLClient(
                name=c, train_arrays=arr, lookback=sc.cfg.lookback, model=copy.deepcopy(self.global_model),
                device=device, seed=seed * 101 + i, lr=float(self.fl.lr), weight_decay=float(self.fl.get("weight_decay", 0.01)),
                batch_size=int(self.fl.batch_size), local_steps=int(self.fl.local_steps),
                grad_clip=self.fl.get("grad_clip", 1.0) if not self.dp_cfg.enabled else None, pos_weight=pw,
                prox_mu=float(self.fl.get("prox_mu", 0.0)),
            )  # fmt: skip
        self.n_total = float(sum(cl.n_samples for cl in self.clients.values()))
        self.budgets = self._setup_privacy() if self.dp_cfg.enabled else {}
        self.val_ds = WindowDataset(sc.arrays(None, "val", self.norm), sc.cfg.lookback)
        self.y_val = self.val_ds.labels().astype(np.float64)
        self.offline = [(str(c), float(a), float(b)) for c, a, b in self.fl.get("offline", [])]
        self.bytes_total = 0
        self.evals: list[dict[str, Any]] = []
        self.best = (-np.inf, copy.deepcopy(self.global_state))
        self._log_event(
            "config", mode=self.fl.mode, algorithm=self.fl.algorithm, clients=names, norm=self.norm,
            n_samples={c: cl.n_samples for c, cl in self.clients.items()},
            n_patients={c: cl.n_patients for c, cl in self.clients.items()}, speed=self.speed,
            model_bytes=self.model_bytes, dp=bool(self.dp_cfg.enabled), budgets=self.budgets,
            offline=self.offline,
        )  # fmt: skip

    # ------------------------------------------------------------------ setup helpers
    def _spec_channels(self) -> int:
        from fedguard.data.features import channel_spec

        f = self.sc.cfg.features
        return channel_spec(f.use_masks, f.use_deltas, list(f.static)).n_channels

    def _setup_privacy(self) -> dict[str, Any]:
        from fedguard.privacy.accounting import calibrate_noise, epsilon_after, planned_steps
        from fedguard.privacy.budgets import allocate

        p = self.dp_cfg
        n_pat = {c: cl.n_patients for c, cl in self.clients.items()}
        targets = allocate(p.budget_rule, float(p.epsilon), n_pat, float(p.get("adaptive_a", 0.55)))
        B, E, R = int(p.logical_batch_size), int(p.local_epochs), int(p.r_max)
        shared_sigma = None
        if p.budget_rule == "equal_noise":  # sigma that uniform eps would give the largest client
            big = max(n_pat, key=n_pat.get)
            q, steps = planned_steps(n_pat[big], B, E, R)
            shared_sigma = calibrate_noise(float(p.epsilon), float(p.delta), q, steps)
        # DIAGNOSTIC ONLY (D32): a fixed noise multiplier (e.g. 0 = the DP pipeline with clipping but no noise).
        # Such runs carry NO privacy guarantee; eps is reported as inf and budgets are not enforced (R_max only).
        override = p.get("noise_multiplier_override")
        out = {}
        for c, cl in self.clients.items():
            q, steps = planned_steps(cl.n_patients, B, E, R)
            if override is not None:
                sigma = float(override)
                target = None
            else:
                sigma = (
                    shared_sigma
                    if shared_sigma is not None
                    else calibrate_noise(float(targets[c]), float(p.delta), q, steps)
                )
                target = targets[c]
            lr = float(p.get("lr", 5e-4))
            if p.get("lr_rule", "fixed") == "sigma_scaled" and sigma > 0:
                # D38: when noise dominates Adam's second moment, DP-Adam acts like DP-SGD with step
                # lr * B_exp / (sigma * C). Keep that effective step at `effective_lr` for every client and eps.
                # sigma, C, q and n_i are public (data-independent), so this is free post-processing.
                lr = float(p.effective_lr) * sigma * float(p.max_grad_norm) / (q * cl.n_patients)
                if p.get("lr_min") is not None:  # floor (non-DP tuned lr): the rule assumes noise dominates, and
                    lr = max(lr, float(p.lr_min))  # without a floor lr -> 0 as sigma -> 0 and high eps under-trains
            setup_dp(cl, target, sigma, float(p.delta), B, R, E, float(p.max_grad_norm),
                     p.get("physical_batch_size"), lr=lr,
                     windows_per_patient=int(p.get("windows_per_patient", 1)),
                     optimizer=str(p.get("optimizer", "adamw")), momentum=float(p.get("momentum", 0.9)))  # fmt: skip
            cl.dp.reset_optimizer = bool(p.get("reset_optimizer", False))
            out[c] = {
                "target_eps": target, "delta": float(p.delta), "noise_multiplier": sigma, "sample_rate": q,
                "planned_steps": steps, "r_max": R, "lr": lr,
                "planned_eps": epsilon_after(sigma, q, steps, float(p.delta)) if sigma > 0 else float("inf"),
                "diagnostic_no_guarantee": override is not None,
            }  # fmt: skip
        return out

    # ------------------------------------------------------------------ helpers
    def _log_event(self, type_: str, t: float = 0.0, **kw: Any) -> None:
        append_jsonl(self.events_path, {"t": round(float(t), 4), "wall": round(time.time() - self.t0_wall, 3),
                                        "type": type_, **kw})  # fmt: skip

    def _eps(self) -> dict[str, float]:
        return {c: round(cl.epsilon(), 5) for c, cl in self.clients.items()}

    def online(self, c: str, t: float) -> bool:
        return not any(cc == c and a <= t < b for cc, a, b in self.offline)

    def offline_during(self, c: str, t0: float, t1: float) -> bool:
        return any(cc == c and a < t1 and b > t0 for cc, a, b in self.offline)

    def next_online(self, c: str, t: float) -> float:
        ends = [b for cc, a, b in self.offline if cc == c and a <= t < b]
        return max(ends) if ends else t

    def duration(self, c: str, samples: int) -> float:
        jitter = max(0.5, 1.0 + float(self.fl.speed_jitter) * float(self.rng.standard_normal()))
        compute = self.speed[c] * samples / 1000.0 * float(self.fl.get("time_per_1k", 1.0)) * jitter
        comm = 2 * self.model_bytes / float(self.fl.get("bandwidth_bytes_per_s", 12.5e6))
        return compute + comm

    def evaluate_global(self, t: float, version: int) -> dict[str, Any]:
        self.global_model.load_state_dict(self.global_state)
        p = loops.predict(self.global_model, self.val_ds, self.device)
        s = M.summary(self.y_val, p)
        rec = {"t": t, "version": version, "val_auroc": s["auroc"], "val_auprc": s["auprc"], "eps": self._eps(),
               "bytes": self.bytes_total}  # fmt: skip
        self.evals.append(rec)
        score = s["auprc"] if np.isfinite(s["auprc"]) else -np.inf
        if score > self.best[0] or not np.isfinite(self.best[0]):
            self.best = (score, copy.deepcopy(self.global_state))
        self._log_event("eval", t, version=version, val_auroc=s["auroc"], val_auprc=s["auprc"], eps=rec["eps"],
                        bytes=self.bytes_total)  # fmt: skip
        return rec

    # ------------------------------------------------------------------ sync
    def run_sync(self) -> None:
        t, version = 0.0, 0
        self.evaluate_global(t, version)
        timeout = float(self.fl.sync_timeout)
        for r in range(1, int(self.fl.rounds) + 1):
            live = [c for c, cl in self.clients.items() if cl.can_participate()]
            if not live:
                self._log_event("all_budgets_exhausted", t, eps=self._eps())
                break
            part = [c for c in live if self.online(c, t)]
            if not part:  # everyone offline: jump to the first reconnection
                t = min(self.next_online(c, t) for c in live)
                part = [c for c in live if self.online(c, t)]
            self._log_event("round_start", t, round=r, version=version, clients=part)
            arrived, lost, arrivals = [], [], []
            for c in part:
                cl = self.clients[c]
                # duration uses the planned local work (known before training); an offline interval that
                # overlaps the job means the update never arrives (and nothing is released, D18)
                planned = self._planned_samples(cl)
                dur = self.duration(c, planned)
                self._log_event("dispatch", t, client=c, version=version)
                self.bytes_total += self.model_bytes
                if self.offline_during(c, t, t + dur):
                    lost.append(c)
                    self._log_event("update_lost", t + dur, client=c, reason="offline")
                    continue
                res = cl.local_train(self.global_state)
                self.bytes_total += self.model_bytes
                arrived.append((c, res))
                arrivals.append(t + dur)
                self._log_event("update_received", t + dur, client=c, version=version, staleness=0,
                                samples=res["samples"], loss=res["loss"], eps=res["eps"])  # fmt: skip
            end = max(arrivals) if arrivals else t
            if lost:
                end = max(end, t + timeout)
                self._log_event("timeout", end, round=r, missing=lost)
            if arrived:
                weights = [res["n"] for _, res in arrived]
                self.global_state = agg.fedavg([res["state"] for _, res in arrived], weights)
                version += 1
                share = {c: w / sum(weights) for (c, _), w in zip(arrived, weights, strict=True)}
                self._log_event("aggregate", end, round=r, version=version, clients=[c for c, _ in arrived],
                                weights=share, eps=self._eps())  # fmt: skip
            t = end
            if r % int(self.fl.eval_every) == 0 or r == int(self.fl.rounds):
                self.evaluate_global(t, version)
            for c, cl in self.clients.items():
                if cl.dp is not None and cl.dp.exhausted and not getattr(cl, "_logged_exhausted", False):
                    cl._logged_exhausted = True  # type: ignore[attr-defined]
                    self._log_event(
                        "budget_exhausted", t, client=c, eps=cl.epsilon(), participations=cl.dp.participations
                    )
        self._finish(t, version)

    def _planned_samples(self, cl: FLClient) -> int:
        if cl.dp is None:
            return cl.local_steps * cl.batch_size
        k = cl.dp.custom.windows_per_patient if cl.dp.custom is not None else 1
        return int(cl.dp.local_epochs * cl.n_patients * k)  # expected windows under Poisson sampling

    # ------------------------------------------------------------------ async
    def run_async(self) -> None:
        t, version, merges = 0.0, 0, 0
        total = int(self.fl.total_updates)
        eval_every = int(self.fl.get("eval_every_updates", len(self.clients)))
        max_tau = int(self.fl.max_staleness)
        heap: list[tuple[float, int, str, str, int, Any]] = []
        seq = 0
        self.evaluate_global(t, version)

        def dispatch(c: str, now: float) -> None:
            nonlocal seq
            cl = self.clients[c]
            if not cl.can_participate():
                return
            if not self.online(c, now):
                heapq.heappush(heap, (self.next_online(c, now), seq, "online", c, version, None))
                seq += 1
                return
            dur = self.duration(c, self._planned_samples(cl))
            self.bytes_total += self.model_bytes
            self._log_event("dispatch", now, client=c, version=version)
            if self.offline_during(c, now, now + dur):
                off_start = min(a for cc, a, b in self.offline if cc == c and a < now + dur and b > now)
                heapq.heappush(heap, (max(off_start, now), seq, "lost", c, version, None))
            else:
                res = cl.local_train(self.global_state)
                self.bytes_total += self.model_bytes
                heapq.heappush(heap, (now + dur, seq, "arrive", c, version, res))
            seq += 1

        for c in self.clients:
            dispatch(c, 0.0)
        while heap and merges < total:
            t, _, kind, c, v_start, res = heapq.heappop(heap)
            if kind == "online":
                self._log_event("online", t, client=c)
                dispatch(c, t)
                continue
            if kind == "lost":
                self._log_event("update_lost", t, client=c, reason="offline")
                heapq.heappush(heap, (self.next_online(c, t + 1e-9), seq, "online", c, version, None))
                seq += 1
                continue
            tau = version - v_start
            cl = self.clients[c]
            if tau > max_tau:
                self._log_event("dropped_stale", t, client=c, staleness=tau)
            else:
                alpha = agg.async_alpha(float(self.fl.alpha0), cl.n_samples, self.n_total, tau, self.fl.staleness_fn,
                                        float(self.fl.staleness_lambda), float(self.fl.staleness_poly_a))  # fmt: skip
                self.global_state = agg.mix(self.global_state, res["state"], alpha)
                version += 1
                merges += 1
                self._log_event("merge", t, client=c, version=version, staleness=tau, weight=alpha,
                                samples=res["samples"], loss=res["loss"], eps=res["eps"])  # fmt: skip
                if merges % eval_every == 0 or merges == total:
                    self.evaluate_global(t, version)
            if cl.dp is not None and cl.dp.exhausted:
                self._log_event(
                    "budget_exhausted", t, client=c, eps=cl.epsilon(), participations=cl.dp.participations
                )
            dispatch(c, t)
        if not self.evals or self.evals[-1]["version"] != version:
            self.evaluate_global(t, version)
        self._finish(t, version)

    def _finish(self, t: float, version: int) -> None:
        self.sim_time, self.version = t, version
        self._log_event("done", t, version=version, eps=self._eps(), bytes=self.bytes_total,
                        best_val_auprc=self.best[0])  # fmt: skip

    def run(self) -> EngineResult:
        if self.fl.mode == "sync":
            self.run_sync()
        elif self.fl.mode == "async":
            self.run_async()
        else:
            raise ValueError(f"unknown fl.mode {self.fl.mode!r}")
        summary = {
            c: {"n_samples": cl.n_samples, "n_patients": cl.n_patients, "participations": len(cl.history),
                "eps": cl.epsilon(), "speed": self.speed[c]}
            for c, cl in self.clients.items()
        }  # fmt: skip
        return EngineResult(
            best_state=self.best[1], last_state=self.global_state, best_val_auprc=float(self.best[0]),
            evals=self.evals, sim_time=self.sim_time, versions=self.version, bytes_total=self.bytes_total,
            client_summary=summary, budgets=self.budgets,
        )  # fmt: skip


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
