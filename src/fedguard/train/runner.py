"""Supervised (non-federated) runs: centralized (pooled) and local-only (one client), any model.

Each run writes to its run dir: config.yaml, meta.json, metrics.csv (per epoch), checkpoints/{best,last}.pt,
preds_val.npz, preds_test.npz (global test set), metrics.json, summary.json, DONE.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from omegaconf import DictConfig

from fedguard.data.scenario import NormMode, Scenario
from fedguard.eval.predictions import Predictions, evaluate
from fedguard.train import loops
from fedguard.train.sklearn_models import fit_predict
from fedguard.utils.io import write_json
from fedguard.utils.logging import MetricsLogger, get_logger
from fedguard.utils.runs import Run
from fedguard.utils.seed import seed_everything


def run_supervised(
    sc: Scenario, cfg: DictConfig, run: Run, train_client: str | None, norm: NormMode
) -> dict[str, Any]:
    """Train on ``train_client`` (None = pooled) and evaluate on the global validation and test sets."""
    log = get_logger("fedguard.train", run.dir / "train.log")
    seed_everything(run.seed)
    t0 = time.time()
    train = sc.dataset(train_client, "train", norm)
    val_own = sc.dataset(train_client, "val", norm)  # early stopping on the training client's own val set
    arr_val_g, arr_test_g = sc.arrays(None, "val", norm), sc.arrays(None, "test", norm)
    from fedguard.data.windows import WindowDataset

    val_g, test_g = WindowDataset(arr_val_g, sc.cfg.lookback), WindowDataset(arr_test_g, sc.cfg.lookback)
    log.info(f"train client={train_client or 'pooled'} norm={norm}: {len(train):,} train / {len(val_own):,} val "
             f"windows; global val {len(val_g):,}, global test {len(test_g):,}")  # fmt: skip
    name = cfg.model.name
    info: dict[str, Any] = {"train_windows": len(train), "train_patients": int(train.a.n_patients)}
    mc: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    if name in ("lr", "lgbm"):
        (p_val_g, p_test), _, extra = fit_predict(name, train, val_own, [val_g, test_g], seed=run.seed,
                                                  max_train=cfg.train.get("max_train_windows", 300_000))  # fmt: skip
        info.update(extra)
    else:
        device = loops.get_device(cfg.train.get("device", "auto"))
        model = loops.build_model(cfg.model, train.a.spec.n_channels, sc.cfg.lookback)
        info["n_params"] = int(sum(p.numel() for p in model.parameters()))
        mlog = MetricsLogger(run.dir, use_wandb=cfg.get("wandb", False))
        hist = loops.fit(
            model, train, val_own, cfg.train, device, run.seed, ckpt_dir=run.checkpoints, log=mlog.log
        )
        mlog.close()
        info.update({k: v for k, v in hist.items() if k != "history"})
        p_val_g = loops.predict(model, val_g, device)
        p_test = loops.predict(model, test_g, device)
        if cfg.get("eval", {}).get("mc_dropout", False):
            T = int(cfg.model.mc_dropout_T)
            for split, ds in (("val", val_g), ("test", test_g)):
                mu, sd = loops.predict_mc(model, ds, device, T=T, seed=run.seed)
                mc[split] = {"p_mc_mean": mu, "p_mc_std": sd}
            info["mc_dropout_T"] = T
    preds_val = Predictions.from_arrays(arr_val_g, p_val_g, **mc["val"])
    preds_test = Predictions.from_arrays(arr_test_g, p_test, **mc["test"])
    preds_val.save(run.dir / "preds_val.npz")
    preds_test.save(run.dir / "preds_test.npz")
    n_boot = int(cfg.get("eval", {}).get("n_boot", 1000))
    metrics = {
        "val": evaluate(preds_val, sc.clients, n_boot=0),
        "test": evaluate(preds_test, sc.clients, n_boot=n_boot, seed=run.seed),
        "info": info,
        "train_client": train_client,
        "norm": norm,
        "wall_clock_s": time.time() - t0,
    }
    write_json(run.dir / "metrics.json", metrics)
    g = metrics["test"]["global"]
    summary = {
        "test_auroc": g["auroc"], "test_auprc": g["auprc"], "test_prevalence": g["prevalence"],
        "test_ece": g["ece"], "test_brier": g["brier"], "wall_clock_s": metrics["wall_clock_s"],
    }  # fmt: skip
    if train_client is not None:
        own = metrics["test"]["per_client"].get(train_client, {})
        summary.update({"own_test_auroc": own.get("auroc"), "own_test_auprc": own.get("auprc")})
    log.info(f"done: {summary}")
    run.mark_done(summary)
    return metrics
