"""M0 scaffold tests: package imports, config composition, run directories, seeding, CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from fedguard.cli import app
from fedguard.config import load_config, validate_data
from fedguard.utils.io import data_dir, runs_dir
from fedguard.utils.runs import config_hash, create_run, find_completed
from fedguard.utils.seed import derived_seed, rng_for, seed_everything


def test_env_dirs_are_isolated(tmp_path):
    assert data_dir() == tmp_path / "data"
    assert runs_dir() == tmp_path / "runs"


def test_compose_experiment_groups():
    cfg = load_config("experiments/centralized")
    assert cfg.data.lookback == 24
    assert cfg.model.d_model == 128
    assert cfg.train.mode == "centralized"
    assert cfg.privacy.enabled is False
    assert "_fast" not in cfg.data and "_fast" not in cfg.model
    assert cfg.fast is False


def test_fast_blocks_are_merged():
    cfg = load_config("experiments/centralized", fast=True)
    assert cfg.data.subset_per_hospital == 150
    assert cfg.model.d_model == 32
    assert cfg.train.epochs == 1
    assert cfg.fast is True


def test_same_group_inheritance_is_flat():
    cfg = load_config("fl/fedprox")
    assert cfg.algorithm == "fedprox" and cfg.prox_mu == 0.01
    assert cfg.local_steps == 500 and cfg.rounds == 40 and "fl" not in cfg
    adaptive = load_config("privacy/adaptive")
    assert adaptive.budget_rule == "adaptive" and adaptive.delta == 1e-5


def test_overrides_and_validation():
    cfg = load_config("experiments/centralized", overrides=["data.unk_policy=separate", "train.lr=0.001"])
    assert cfg.train.lr == 0.001
    assert validate_data(cfg).unk_policy == "separate"
    bad = load_config("experiments/centralized", overrides=["data.unk_policy=drop"])
    with pytest.raises(Exception, match="unk_policy"):
        validate_data(bad)


def test_run_dir_provenance(tmp_path):
    cfg = load_config("experiments/centralized", fast=True)
    run = create_run("unit_test", seed=3, cfg=cfg)
    assert run.dir.parent == tmp_path / "runs" / "unit_test"
    assert run.dir.name.endswith("_3")
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["seed"] == 3 and "torch" in meta["versions"] and "commit" in meta["git"]
    assert meta["config_hash"] == config_hash(cfg)
    assert (run.dir / "config.yaml").exists()
    assert find_completed("unit_test", 3) is None
    run.mark_done({"val_auroc": None})
    assert find_completed("unit_test", 3) == run.dir
    assert find_completed("unit_test", 3, config_hash=config_hash(cfg)) == run.dir
    assert find_completed("unit_test", 3, config_hash="nope") is None


def test_seeding_is_deterministic():
    import torch

    seed_everything(7)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.equal(a[1], b[1])
    assert derived_seed(42, "A_MICU") == derived_seed(42, "A_MICU") != derived_seed(42, "B_MICU")
    assert rng_for(1, "x").integers(1 << 30) == rng_for(1, "x").integers(1 << 30)


def test_cli_info_and_empty_report(tmp_path, monkeypatch):
    runner = CliRunner()
    res = runner.invoke(app, ["info"])
    assert res.exit_code == 0, res.output
    assert "data_dir" in json.loads(res.stdout)
    # with no runs at all the report must still work and say "not run yet" (never invent numbers)
    monkeypatch.setenv("FEDGUARD_RESULTS_DIR", str(tmp_path / "results"))
    res = runner.invoke(app, ["report", "--fast"])
    assert res.exit_code == 0, res.output
    assert "not run yet" in res.stdout
