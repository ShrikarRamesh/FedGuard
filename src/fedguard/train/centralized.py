"""Centralized training: pool every client's training patients, pooled normalisation (upper bound)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from fedguard.data.scenario import Scenario
from fedguard.train.runner import run_supervised
from fedguard.utils.logging import get_logger
from fedguard.utils.runs import config_hash, create_run, find_completed


def run_centralized(
    sc: Scenario, cfg: DictConfig, seed: int, experiment: str, resume: bool = True
) -> dict[str, Any] | Path:
    """One seed. If an identical completed run exists (same config hash) it is skipped and its dir returned."""
    cfg.seed = seed
    done = find_completed(experiment, seed, config_hash(cfg)) if resume else None
    if done is not None:
        get_logger().info(f"skip: {experiment} seed {seed} already done in {done}")
        return done
    run = create_run(experiment, seed, cfg)
    return run_supervised(sc, cfg, run, train_client=None, norm="pooled")
