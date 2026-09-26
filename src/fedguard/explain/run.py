"""Global explanations for one finished run -> explain.json + plots.

* Global feature importance: mean |IG| per variable (summed over the 24 h window) over a seeded sample of
  test windows (half positive, half negative hours).
* Temporal importance: mean |IG| per hour-of-window, and mean attention-rollout per hour-of-window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedguard.data.windows import WindowDataset  # noqa: E402
from fedguard.eval.run_artifacts import load_model, run_config, run_norm, run_scenario  # noqa: E402
from fedguard.explain.attention import rollout  # noqa: E402
from fedguard.explain.integrated_gradients import integrated_gradients, per_variable  # noqa: E402
from fedguard.train import loops  # noqa: E402
from fedguard.utils.io import write_json  # noqa: E402


def sample_windows(ds: WindowDataset, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = ds.labels()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = min(n // 2, len(pos))
    return np.sort(
        np.concatenate(
            [rng.choice(pos, k, replace=False), rng.choice(neg, min(n - k, len(neg)), replace=False)]
        )
    )


def run_explain(run_dir: Path, n_windows: int = 1000, seed: int = 0) -> dict[str, Any]:
    cfg = run_config(run_dir)
    sc = run_scenario(cfg)
    device = loops.get_device("auto")
    arr = sc.arrays(None, "test", run_norm(run_dir))
    ds = WindowDataset(arr, sc.cfg.lookback)
    model = load_model(run_dir, cfg, arr.spec.n_channels, device)
    idx = sample_windows(ds, n_windows, seed)
    x, m, y = ds.gather(idx)
    xt, mt = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)
    attr = integrated_gradients(model, xt, mt, n_steps=50, internal_batch_size=1024).cpu().numpy()
    names, pv = per_variable(attr, arr.spec)
    imp = np.abs(pv.sum(1)).mean(0)  # |sum over hours| per variable
    imp_pos = np.abs(pv[y == 1].sum(1)).mean(0) if (y == 1).any() else imp * np.nan
    hour_ig = np.abs(pv).sum(-1).mean(0)
    ro = rollout(model, xt, mt).mean(0) if cfg.model.name == "patchtst" else None
    order = np.argsort(imp)[::-1]
    out = {
        "variables": names, "importance": imp.tolist(), "importance_pos": imp_pos.tolist(),
        "ranking": [names[i] for i in order], "hour_ig": hour_ig.tolist(),
        "hour_rollout": None if ro is None else ro.tolist(), "n_windows": int(len(idx)), "n_pos": int(y.sum()),
    }  # fmt: skip
    write_json(Path(run_dir) / "explain.json", out)
    top = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.barh([names[i] for i in top], imp[top], color="#1F6E8C")
    ax.set_xlabel("mean |IG attribution| (logit)")
    ax.set_title("Global feature importance (test windows)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "feature_importance.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    hrs = np.arange(-len(hour_ig) + 1, 1)
    ax.plot(hrs, hour_ig / hour_ig.sum(), color="#1F6E8C", label="IG (normalised)")
    if ro is not None:
        ax.plot(hrs, ro, color="#C8413A", label="attention rollout")
    ax.set_xlabel("hour relative to prediction time")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "temporal_importance.png", dpi=150)
    plt.close(fig)
    return out
