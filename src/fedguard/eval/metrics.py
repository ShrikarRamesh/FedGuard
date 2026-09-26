"""Discrimination and calibration metrics with patient-level bootstrap CIs.

Every metric returns NaN (never a made-up number) when it is not computable, e.g. AUROC on a set
with a single class. AUPRC is always reported together with the prevalence of the evaluated set.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _both_classes(y: np.ndarray) -> bool:
    return len(y) > 0 and 0 < y.sum() < len(y)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if _both_classes(y) else float("nan")


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (step-wise AUPRC). Compare against ``prevalence`` (the random-classifier AUPRC)."""
    return float(average_precision_score(y, p)) if _both_classes(y) else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with ``n_bins`` equal-width bins (weighted by bin count)."""
    if len(y) == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins)
    acc = np.bincount(bins, weights=y, minlength=n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(acc[nz] - conf[nz])) / len(y))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, list[float]]:
    """Per-bin mean predicted probability, observed frequency and count (for reliability diagrams)."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins)
    acc = np.bincount(bins, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "bin_lower": (np.arange(n_bins) / n_bins).tolist(),
            "mean_pred": (conf / cnt).tolist(),
            "frac_pos": (acc / cnt).tolist(),
            "count": cnt.tolist(),
        }


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier,
    "ece": ece,
}


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Point estimates of all metrics plus prevalence and counts."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out = {k: f(y, p) for k, f in METRICS.items()}
    out["prevalence"] = float(y.mean()) if len(y) else float("nan")
    out["n_rows"] = int(len(y))
    out["n_pos"] = int(y.sum())
    return out


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    patient: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    metrics: tuple[str, ...] = ("auroc", "auprc"),
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling **patients** with replacement (all rows of a drawn patient are kept).

    ``patient`` is a per-row patient identifier. Resamples where a metric is undefined are skipped; if
    fewer than half are valid the CI is reported as (nan, nan).
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(patient, kind="stable")
    y, p, patient = y[order], p[order], np.asarray(patient)[order]
    _, starts, counts = np.unique(patient, return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {m: [] for m in metrics}
    n_pat = len(starts)
    for _ in range(n_boot):
        draw = rng.integers(0, n_pat, n_pat)
        c = counts[draw]
        # rows of drawn patients: start of each drawn patient repeated, plus position within patient
        within = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        idx = np.repeat(starts[draw], c) + within
        for m in metrics:
            vals[m].append(METRICS[m](y[idx], p[idx]))
    out = {}
    for m, v in vals.items():
        arr = np.asarray(v)
        arr = arr[np.isfinite(arr)]
        if len(arr) < n_boot / 2:
            out[m] = (float("nan"), float("nan"))
        else:
            out[m] = (float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2)))
    return out
