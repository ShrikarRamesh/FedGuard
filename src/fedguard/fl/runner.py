"""Run one federated experiment end to end: engine -> best-on-validation global model -> test predictions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from fedguard.data.scenario import Scenario
from fedguard.data.windows import WindowDataset
from fedguard.eval.predictions import Predictions, evaluate
from fedguard.fl.engine import FLEngine
from fedguard.train import loops
from fedguard.utils.io import write_json
from fedguard.utils.logging import get_logger
from fedguard.utils.runs import config_hash, create_run, find_completed
from fedguard.utils.seed import seed_everything


def eval_checkpoints(run_dir: Path, sc: Scenario, cfg: DictConfig, n_boot: int = 1000) -> dict[str, Any]:
    """Score every ``checkpoints/round_*.pt`` of a Flower run on the global validation set, and the final model
    on the global test set, with exactly the in-house evaluation code (Flower cross-check, D19)."""
    from fedguard.eval import metrics as M

    device = loops.get_device("auto")
    arr_val, arr_test = sc.arrays(None, "val", "site"), sc.arrays(None, "test", "site")
    val_ds, test_ds = WindowDataset(arr_val, sc.cfg.lookback), WindowDataset(arr_test, sc.cfg.lookback)
    model = loops.build_model(cfg.model, arr_val.spec.n_channels, sc.cfg.lookback).to(device)
    evals = []
    y_val = val_ds.labels().astype(float)
    ckpts = sorted((run_dir / "checkpoints").glob("round_*.pt"))
    best = (-float("inf"), None)
    if not ckpts:
        raise FileNotFoundError(f"no round checkpoints in {run_dir / 'checkpoints'}")
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True)["model"])
        s = M.summary(y_val, loops.predict(model, val_ds, device))
        rnd = int(ck.stem.split("_")[1])
        evals.append({"round": rnd, "val_auroc": s["auroc"], "val_auprc": s["auprc"]})
        if s["auprc"] > best[0]:  # NaN never wins
            best = (s["auprc"], ck)
    chosen = best[1] if best[1] is not None else ckpts[-1]  # val AUPRC undefined everywhere -> last model
    model.load_state_dict(torch.load(chosen, map_location=device, weights_only=True)["model"])
    pt = Predictions.from_arrays(arr_test, loops.predict(model, test_ds, device))
    pt.save(run_dir / "preds_test.npz")
    out = {
        "evals": evals,
        "best_round_ckpt": chosen.name,
        "test": evaluate(pt, sc.clients, n_boot=n_boot, seed=0),
    }
    write_json(run_dir / "metrics.json", out)
    return out


def run_fl(
    sc: Scenario, cfg: DictConfig, seed: int, experiment: str, resume: bool = True
) -> dict[str, Any] | Path:
    cfg.seed = seed
    done = find_completed(experiment, seed, config_hash(cfg)) if resume else None
    if done is not None:
        get_logger().info(f"skip: {experiment} seed {seed} already done in {done}")
        return done
    run = create_run(experiment, seed, cfg)
    log = get_logger("fedguard.fl", run.dir / "train.log")
    seed_everything(seed)
    t0 = time.time()
    device = loops.get_device(cfg.fl.get("device", "auto"))
    engine = FLEngine(sc, cfg, run.dir, seed, device)
    log.info(f"FL {cfg.fl.algorithm} mode={cfg.fl.mode} dp={cfg.privacy.enabled} norm={engine.norm} "
             f"clients={list(engine.clients)}")  # fmt: skip
    res = engine.run()
    torch.save({"model": res.best_state}, run.checkpoints / "best.pt")
    torch.save({"model": res.last_state}, run.checkpoints / "last.pt")
    model = engine.global_model
    model.load_state_dict(res.best_state)
    arr_val, arr_test = sc.arrays(None, "val", engine.norm), sc.arrays(None, "test", engine.norm)
    val_ds, test_ds = WindowDataset(arr_val, sc.cfg.lookback), WindowDataset(arr_test, sc.cfg.lookback)
    mc: dict[str, dict] = {"val": {}, "test": {}}
    if cfg.get("eval", {}).get("mc_dropout", False):
        for split, ds in (("val", val_ds), ("test", test_ds)):
            mu, sd = loops.predict_mc(model, ds, device, T=int(cfg.model.mc_dropout_T), seed=seed)
            mc[split] = {"p_mc_mean": mu, "p_mc_std": sd}
    pv = Predictions.from_arrays(arr_val, loops.predict(model, val_ds, device), **mc["val"])
    pt = Predictions.from_arrays(arr_test, loops.predict(model, test_ds, device), **mc["test"])
    pv.save(run.dir / "preds_val.npz")
    pt.save(run.dir / "preds_test.npz")
    metrics = {
        "val": evaluate(pv, sc.clients, n_boot=0),
        "test": evaluate(pt, sc.clients, n_boot=int(cfg.get("eval", {}).get("n_boot", 1000)), seed=seed),
        "fl": {"sim_time": res.sim_time, "versions": res.versions, "bytes_total": res.bytes_total,
               "model_bytes": engine.model_bytes, "best_val_auprc": res.best_val_auprc, "evals": res.evals,
               "clients": res.client_summary, "budgets": res.budgets},  # fmt: skip
        "norm": engine.norm,
        "wall_clock_s": time.time() - t0,
    }
    write_json(run.dir / "metrics.json", metrics)
    g = metrics["test"]["global"]
    summary = {
        "test_auroc": g["auroc"], "test_auprc": g["auprc"], "test_prevalence": g["prevalence"], "test_ece": g["ece"],
        "test_brier": g["brier"], "sim_time": res.sim_time, "bytes_total": res.bytes_total,
        "eps": {c: s["eps"] for c, s in res.client_summary.items()}, "wall_clock_s": metrics["wall_clock_s"],
    }  # fmt: skip
    log.info(f"done: {summary}")
    run.mark_done(summary)
    return metrics
