"""Shared helpers for the Flower ServerApp / ClientApp (Flower 1.38 Message API).

The Flower app reuses FedGuard's own config, model and local-training code so that Flower FedAvg and the
in-house sync FedAvg are the same algorithm (cross-check, D19). Run config keys (pyproject.toml):

    experiment        config path, e.g. "experiments/fedavg"
    seed              int
    fast              bool (tiny model / subset)
    num-server-rounds int
    out-dir           where the server writes events.jsonl + per-round checkpoints ("" = runs/flower/<ts>)

Node config keys (``flower-supernode --node-config``):

    client     node name, e.g. "A_MICU"   (the SuperNode trains ONLY on this node's patients)
    data-dir   processed data dir holding this node's data (default: <FEDGUARD_DATA_DIR>/processed[_fast])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from fedguard.config import experiment_config, validate_data
from fedguard.data.scenario import Scenario
from fedguard.utils.io import data_dir


def cfg_from_run_config(rc: dict[str, Any]) -> DictConfig:
    fast = bool(rc.get("fast", False))
    cfg = experiment_config(str(rc.get("experiment", "experiments/fedavg")), fast)
    cfg.seed = int(rc.get("seed", 0))
    if "num-server-rounds" in rc:
        cfg.fl.rounds = int(rc["num-server-rounds"])
    return cfg


def scenario_for(cfg: DictConfig, node_cfg: dict[str, Any]) -> Scenario:
    dcfg = validate_data(cfg)
    default = data_dir() / (dcfg.processed_subdir + ("_fast" if cfg.fast else ""))
    return Scenario.load(Path(str(node_cfg.get("data-dir", default))), dcfg)
