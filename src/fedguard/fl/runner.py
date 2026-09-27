"""Run one federated experiment end to end: engine -> best-on-validation global model -> test predictions."""

from __future__ import annotations

import json
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
    loops.HEARTBEAT = run.dir / "heartbeat"  # progress signal for scripts/watchdog.py
    t0 = time.time()
    device = loops.get_device(cfg.fl.get("device", "auto"))
    engine = FLEngine(sc, cfg, run.dir, seed, device)
    log.info(f"FL {cfg.fl.algorithm} mode={cfg.fl.mode} dp={cfg.privacy.enabled} norm={engine.norm} "
             f"clients={list(engine.clients)}")  # fmt: skip
    res = engine.run()
    torch.save({"model": res.best_state}, run.checkpoints / "best.pt")
    torch.save({"model": res.last_state}, run.checkpoints / "last.pt")
    return _evaluate_and_finish(run, sc, cfg, seed, engine.global_model, device, engine.norm, res,
                                engine.model_bytes, t0, log)  # fmt: skip


def finalize_fl(run_dir: Path, sc: Scenario, cfg: DictConfig) -> dict[str, Any]:
    """Finish a run whose training completed (``done`` event logged, ``checkpoints/best.pt`` saved) but whose process
    died in the post-training evaluation. Runs exactly the post-training code of :func:`run_fl` on the saved best
    model; the engine result is rebuilt from ``events.jsonl`` (every field is logged there). D37."""
    from fedguard.fl.engine import EngineResult
    from fedguard.utils.io import read_json
    from fedguard.utils.runs import DONE_MARKER, Run

    if (run_dir / DONE_MARKER).exists():
        raise FileExistsError(f"{run_dir} is already done")
    meta = read_json(run_dir / "meta.json")
    if meta["config_hash"] != config_hash(cfg):
        raise ValueError(f"config in {run_dir} does not match its recorded hash")
    ev = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    conf = next(e for e in ev if e["type"] == "config")
    done = [e for e in ev if e["type"] == "done"]
    if not done:
        raise RuntimeError(f"{run_dir}: training did not finish (no 'done' event); rerun it instead")
    fin = done[-1]
    if conf["mode"] != "async":
        raise NotImplementedError("finalize supports async runs (participations = dispatches); rerun sync runs")
    # async clients train when dispatched (updates still in flight at the end count as participations)
    parts = {c: sum(e["type"] == "dispatch" and e.get("client") == c for e in ev) for c in conf["clients"]}
    eps = {c: _exact_eps(conf.get("budgets", {}).get(c), parts[c], cfg, fin["eps"][c]) for c in conf["clients"]}
    evals = [{k: e[k] for k in ("t", "version", "val_auroc", "val_auprc", "eps", "bytes")}
             for e in ev if e["type"] == "eval"]  # fmt: skip
    best = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)["model"]
    res = EngineResult(
        best_state=best, last_state={}, best_val_auprc=float(fin["best_val_auprc"]), evals=evals,
        sim_time=float(fin["t"]), versions=int(fin["version"]), bytes_total=int(fin["bytes"]),
        client_summary={c: {"n_samples": conf["n_samples"][c], "n_patients": conf["n_patients"][c],
                            "participations": parts[c], "eps": eps[c], "speed": conf["speed"][c]}
                        for c in conf["clients"]},
        budgets=conf.get("budgets", {}),
    )  # fmt: skip
    run = Run(run_dir, meta["experiment"], int(meta["seed"]))
    log = get_logger("fedguard.fl", run_dir / "train.log")
    log.info(f"finalize: training finished earlier (version {res.versions}); evaluating checkpoints/best.pt (D37)")
    seed_everything(run.seed)
    loops.HEARTBEAT = run_dir / "heartbeat"
    device = loops.get_device(cfg.fl.get("device", "auto"))
    arr = sc.arrays(None, "val", conf["norm"])
    model = loops.build_model(cfg.model, arr.spec.n_channels, sc.cfg.lookback).to(device)
    return _evaluate_and_finish(run, sc, cfg, run.seed, model, device, conf["norm"], res, int(conf["model_bytes"]),
                                time.time(), log, training_wall_s=float(fin["wall"]))  # fmt: skip


def _exact_eps(budget: dict | None, participations: int, cfg: DictConfig, logged: float) -> float:
    """Total RDP epsilon of a client after ``participations`` local jobs, recomputed exactly (the ``done`` event
    rounds it to 1e-5). Each job is ``planned_steps / r_max`` Poisson steps at the calibrated noise, which is the
    history the client's Opacus accountant holds; the result is checked against the logged value."""
    from fedguard.privacy.accounting import epsilon_after

    if budget is None or participations == 0:
        return 0.0 if logged == 0 else float(logged)
    if int(cfg.privacy.get("windows_per_patient", 1)) != 1 or budget["noise_multiplier"] <= 0:
        raise NotImplementedError("finalize: exact epsilon only for the standard patient-level Opacus accountant")
    steps = budget["planned_steps"] * participations // budget["r_max"]
    e = epsilon_after(budget["noise_multiplier"], budget["sample_rate"], steps, budget["delta"])
    if abs(e - logged) > 1e-5:
        raise RuntimeError(f"recomputed epsilon {e} != logged {logged}")
    return e


def _evaluate_and_finish(run, sc: Scenario, cfg: DictConfig, seed: int, model: torch.nn.Module, device, norm: str,
                         res, model_bytes: int, t0: float, log, training_wall_s: float | None = None,
                         ) -> dict[str, Any]:  # fmt: skip
    """Post-training: best model -> (MC) predictions on val/test -> metrics.json, summary.json, DONE."""
    model.load_state_dict(res.best_state)
    arr_val, arr_test = sc.arrays(None, "val", norm), sc.arrays(None, "test", norm)
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
               "model_bytes": model_bytes, "best_val_auprc": res.best_val_auprc, "evals": res.evals,
               "clients": res.client_summary, "budgets": res.budgets},  # fmt: skip
        "norm": norm,
        "wall_clock_s": time.time() - t0 + (training_wall_s or 0.0),
    }
    if training_wall_s is not None:
        metrics["finalized_from_checkpoint"] = {"training_wall_s": training_wall_s,
                                                "eval_wall_s": metrics["wall_clock_s"] - training_wall_s}  # fmt: skip
    write_json(run.dir / "metrics.json", metrics)
    g = metrics["test"]["global"]
    summary = {
        "test_auroc": g["auroc"], "test_auprc": g["auprc"], "test_prevalence": g["prevalence"], "test_ece": g["ece"],
        "test_brier": g["brier"], "sim_time": res.sim_time, "bytes_total": res.bytes_total,
        "eps": {c: s["eps"] for c, s in res.client_summary.items()}, "wall_clock_s": metrics["wall_clock_s"],
    }  # fmt: skip
    if training_wall_s is not None:
        summary["finalized_from_checkpoint"] = True
    log.info(f"done: {summary}")
    # a finalize process never trained, so its peak GPU memory is not the run's training peak
    run.mark_done(summary, gpu_key="peak_gpu_mem_mib" if training_wall_s is None else "peak_gpu_mem_mib_finalize")
    return metrics
