"""Load artefacts of a finished run: its config, scenario, trained model and predictions."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from fedguard.config import validate_data
from fedguard.data.scenario import NormMode, Scenario
from fedguard.eval.predictions import Predictions
from fedguard.train import loops
from fedguard.utils.io import data_dir, read_json


def run_config(run_dir: Path) -> DictConfig:
    return OmegaConf.load(Path(run_dir) / "config.yaml")  # type: ignore[return-value]


def run_scenario(cfg: DictConfig) -> Scenario:
    dcfg = validate_data(cfg)
    return Scenario.load(data_dir() / (dcfg.processed_subdir + ("_fast" if cfg.get("fast") else "")), dcfg)


def run_norm(run_dir: Path) -> NormMode:
    """Normalisation used by the run (recorded in metrics.json)."""
    return read_json(Path(run_dir) / "metrics.json")["norm"]  # type: ignore[no-any-return]


def load_model(run_dir: Path, cfg: DictConfig, n_channels: int, device: torch.device) -> torch.nn.Module:
    """Best-on-validation model of a run (supervised or federated)."""
    model = loops.build_model(cfg.model, n_channels, int(cfg.data.lookback)).to(device)
    ck = torch.load(Path(run_dir) / "checkpoints" / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    return model.eval()


def load_preds(run_dir: Path, split: str) -> Predictions:
    return Predictions.load(Path(run_dir) / f"preds_{split}.npz")
