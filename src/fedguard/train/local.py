"""Local-only training: each client trains alone on its own data with its own stats (lower bound).

Each local model is evaluated on (a) its own client's test set (``per_client[<node>]``) and (b) the global
test set, where every patient is standardised with its own site's training stats (D15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from fedguard.data.scenario import Scenario
from fedguard.train.runner import run_supervised
from fedguard.utils.logging import get_logger
from fedguard.utils.runs import config_hash, create_run, find_completed


def run_local(
    sc: Scenario, cfg: DictConfig, seed: int, node: str, experiment: str, resume: bool = True
) -> dict[str, Any] | Path:
    if node not in sc.client_names:
        raise ValueError(f"unknown node {node!r}; available: {sc.client_names}")
    cfg.seed = seed
    cfg.node = node
    exp = f"{experiment}/{node}"
    done = find_completed(exp, seed, config_hash(cfg)) if resume else None
    if done is not None:
        get_logger().info(f"skip: {exp} seed {seed} already done in {done}")
        return done
    run = create_run(exp, seed, cfg)
    return run_supervised(sc, cfg, run, train_client=node, norm="site")
