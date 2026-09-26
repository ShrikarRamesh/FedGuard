"""Aggregation rules: FedAvg (weighted mean), and staleness-weighted asynchronous mixing (FedGuard / FedAsync).

Async mixing (FedGuard): w_global <- (1 - a_i) * w_global + a_i * w_client with
    a_i = alpha0 * (n_i / N) * s(tau),  s(tau) = exp(-lambda * tau)           ("exp", FedGuard default)
                                        s(tau) = (1 + tau) ** (-a)            ("poly", FedAsync; Xie et al. 2019)
tau = server version now - version the client started from. Updates with tau > max_staleness are dropped.
"""

from __future__ import annotations

import math

import torch

State = dict[str, torch.Tensor]


def fedavg(states: list[State], weights: list[float]) -> State:
    """Weighted average of state dicts (weights normalised to sum to 1). Non-float entries are copied."""
    if not states:
        raise ValueError("no states to aggregate")
    w = torch.tensor(weights, dtype=torch.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"invalid weights {weights}")
    w = w / w.sum()
    out: State = {}
    for k, v0 in states[0].items():
        if not torch.is_floating_point(v0):
            out[k] = v0.clone()
            continue
        acc = torch.zeros_like(v0, dtype=torch.float64)
        for wi, s in zip(w, states, strict=True):
            acc += wi * s[k].to(torch.float64)
        out[k] = acc.to(v0.dtype)
    return out


def staleness_factor(tau: int, fn: str = "exp", lam: float = 0.35, poly_a: float = 0.5) -> float:
    """Monotonically non-increasing weight in the staleness tau >= 0 (1 at tau = 0)."""
    if tau < 0:
        raise ValueError("staleness cannot be negative")
    if fn == "exp":
        return math.exp(-lam * tau)
    if fn == "poly":
        return (1.0 + tau) ** (-poly_a)
    if fn == "const":
        return 1.0
    raise ValueError(f"unknown staleness function {fn!r}")


def async_alpha(alpha0: float, n_i: float, n_total: float, tau: int, fn: str = "exp", lam: float = 0.35,
                poly_a: float = 0.5) -> float:  # fmt: skip
    """Mixing weight a_i in [0, 1]."""
    a = alpha0 * (n_i / n_total) * staleness_factor(tau, fn, lam, poly_a)
    return float(min(max(a, 0.0), 1.0))


def mix(global_state: State, client_state: State, alpha: float) -> State:
    """(1 - alpha) * global + alpha * client for float entries."""
    return {
        k: ((1 - alpha) * g + alpha * client_state[k]) if torch.is_floating_point(g) else g.clone()
        for k, g in global_state.items()
    }


def state_bytes(state: State) -> int:
    """Bytes needed to transmit a state dict as raw tensors."""
    return int(sum(v.numel() * v.element_size() for v in state.values()))
