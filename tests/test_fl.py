"""FL tests (M4): aggregation, staleness, deterministic simulated clock, offline clients."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fedguard.config import DataConfig, experiment_config
from fedguard.data.physionet2019 import process
from fedguard.data.scenario import Scenario
from fedguard.fl import aggregators as agg
from fedguard.fl.engine import FLEngine
from tests.fixtures_data import write_fixture


def test_fedavg_identical_models_is_identity():
    s = {"w": torch.randn(3, 4), "b": torch.randn(4), "step": torch.tensor(7)}
    out = agg.fedavg([s, s, s], [1, 5, 2])
    torch.testing.assert_close(out["w"], s["w"])
    torch.testing.assert_close(out["b"], s["b"])
    assert out["step"] == 7


def test_fedavg_weights_normalised():
    a, b = {"w": torch.zeros(2)}, {"w": torch.ones(2)}
    torch.testing.assert_close(agg.fedavg([a, b], [1, 3])["w"], torch.full((2,), 0.75))
    torch.testing.assert_close(agg.fedavg([a, b], [10, 30])["w"], torch.full((2,), 0.75))
    with pytest.raises(ValueError):
        agg.fedavg([a, b], [0, 0])


def test_staleness_weights_decrease():
    for fn in ("exp", "poly"):
        w = [agg.staleness_factor(t, fn) for t in range(10)]
        assert w[0] == 1.0 and all(x > y for x, y in zip(w, w[1:], strict=False))
    a = [agg.async_alpha(1.0, 100, 400, t) for t in range(5)]
    assert a[0] == pytest.approx(0.25) and all(x > y for x, y in zip(a, a[1:], strict=False))
    g, c = {"w": torch.zeros(3)}, {"w": torch.ones(3)}
    torch.testing.assert_close(agg.mix(g, c, 0.2)["w"], torch.full((3,), 0.2))


@pytest.fixture(scope="module")
def tiny_scenario(tmp_path_factory):
    d = tmp_path_factory.mktemp("fl")
    write_fixture(d / "raw", n_per_hospital=40)
    cfg = DataConfig()
    process(cfg, d / "raw", d / "processed", workers=1)
    return Scenario.load(d / "processed", cfg)


def tiny_cfg(exp: str, **over):
    ov = ["model.d_model=16", "model.n_layers=1", "model.n_heads=2", "model.d_ff=16", "model.head_hidden=8",
          "fl.local_steps=3", "fl.batch_size=16", "fl.rounds=3", "fl.device=cpu", "eval.mc_dropout=false"]  # fmt: skip
    ov += [f"{k}={v}" for k, v in over.items()]
    return experiment_config(f"experiments/{exp}", fast=True, overrides=ov)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_engine(sc, cfg, tmp, seed=0):
    eng = FLEngine(sc, cfg, tmp, seed, torch.device("cpu"))
    res = eng.run()
    return eng, res, events(tmp / "events.jsonl")


def test_simulated_clock_is_deterministic(tiny_scenario, tmp_path):
    cfg = tiny_cfg("fedguard_async_nodp", **{"fl.total_updates": 6})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, r1, e1 = run_engine(tiny_scenario, cfg, tmp_path / "a")
    _, r2, e2 = run_engine(tiny_scenario, cfg, tmp_path / "b")
    strip = [[(e["t"], e["type"], e.get("client"), e.get("staleness")) for e in ev] for ev in (e1, e2)]
    assert strip[0] == strip[1]
    for k in r1.last_state:
        torch.testing.assert_close(r1.last_state[k], r2.last_state[k])
    merges = [e for e in e1 if e["type"] == "merge"]
    assert len(merges) == 6 and all(e["staleness"] >= 0 for e in merges)
    assert any(e["staleness"] > 0 for e in merges)  # heterogeneous speeds -> some stale updates
    assert [e["t"] for e in e1] == sorted(e["t"] for e in e1)


def test_sync_rounds_and_weights(tiny_scenario, tmp_path):
    eng, res, ev = run_engine(tiny_scenario, tiny_cfg("fedavg"), tmp_path)
    aggs = [e for e in ev if e["type"] == "aggregate"]
    assert len(aggs) == 3 and res.versions == 3
    for a in aggs:
        assert sum(a["weights"].values()) == pytest.approx(1.0)
        exp = {c: eng.clients[c].n_samples for c in a["clients"]}
        tot = sum(exp.values())
        assert all(a["weights"][c] == pytest.approx(exp[c] / tot) for c in a["clients"])
    # the round ends when the slowest client arrives (sync waits for stragglers)
    rec = [e for e in ev if e["type"] == "update_received" and e["version"] == 0]
    assert aggs[0]["t"] == pytest.approx(max(e["t"] for e in rec), abs=1e-3)
    assert res.bytes_total == 2 * 3 * len(eng.clients) * eng.model_bytes


def test_sync_offline_client_times_out(tiny_scenario, tmp_path):
    off = "[[B_SICU,0.0,1.0e9]]"
    cfg = tiny_cfg("fedavg", **{"fl.offline": off, "fl.sync_timeout": 500.0})
    _, res, ev = run_engine(tiny_scenario, cfg, tmp_path)
    aggs = [e for e in ev if e["type"] == "aggregate"]
    assert all("B_SICU" not in a["clients"] for a in aggs) and len(aggs) == 3
    # offline for the whole run -> never dispatched, no timeout needed
    assert not any(e["type"] == "dispatch" and e["client"] == "B_SICU" for e in ev)
    # goes offline mid-job in round 1 -> update lost, server waits for the timeout
    # tiny jobs last ~0.05 simulated s, so an outage starting at 0.01 s overlaps the first job
    cfg = tiny_cfg("fedavg", **{"fl.offline": "[[B_SICU,0.01,2.0]]", "fl.sync_timeout": 500.0})
    (tmp_path / "x").mkdir()
    _, _, ev = run_engine(tiny_scenario, cfg, tmp_path / "x")
    assert any(e["type"] == "update_lost" and e["client"] == "B_SICU" for e in ev)
    t_out = [e for e in ev if e["type"] == "timeout"]
    assert t_out and t_out[0]["t"] == pytest.approx(500.0, abs=1e-3)


def test_async_offline_client_rejoins(tiny_scenario, tmp_path):
    cfg = tiny_cfg("fedguard_async_nodp", **{"fl.total_updates": 30, "fl.offline": "[[A_MICU,0.01,0.3]]"})
    _, _, ev = run_engine(tiny_scenario, cfg, tmp_path)
    lost = [e for e in ev if e["type"] == "update_lost" and e["client"] == "A_MICU"]
    back = [e for e in ev if e["type"] == "online" and e["client"] == "A_MICU"]
    assert lost and back and back[0]["t"] == pytest.approx(0.3)
    assert not [e for e in ev if e["type"] == "merge" and e["client"] == "A_MICU" and 0.01 <= e["t"] < 0.3]
    assert [e for e in ev if e["type"] == "merge" and e["client"] == "A_MICU" and e["t"] > 0.3]


def test_stale_updates_dropped(tiny_scenario, tmp_path):
    cfg = tiny_cfg("fedguard_async_nodp", **{"fl.total_updates": 12, "fl.max_staleness": 0})
    _, _, ev = run_engine(tiny_scenario, cfg, tmp_path)
    assert all(e["staleness"] == 0 for e in ev if e["type"] == "merge")
    assert any(e["type"] == "dropped_stale" for e in ev)


def test_fl_runner_writes_artifacts(tiny_scenario, tmp_path):
    from fedguard.fl.runner import run_fl

    cfg = tiny_cfg("fedprox")
    m = run_fl(tiny_scenario, cfg, 0, "unit_fedprox", resume=False)
    assert "global" in m["test"] and np.isfinite(m["test"]["global"]["brier"])
    run_dir = next((tmp_path / "runs" / "unit_fedprox").iterdir())
    for f in (
        "events.jsonl",
        "preds_test.npz",
        "preds_val.npz",
        "metrics.json",
        "DONE",
        "checkpoints/best.pt",
    ):
        assert (run_dir / f).exists(), f


@pytest.mark.parametrize("exp", ["fedguard_async_nodp", "fedguard"])
def test_finalize_reproduces_post_training_eval(tiny_scenario, tmp_path, exp):
    """D37: a run that died after training can be finished from events.jsonl + best.pt, with identical outputs."""
    from omegaconf import OmegaConf

    from fedguard.fl.runner import finalize_fl, run_fl
    from fedguard.utils.io import read_json

    cfg = tiny_cfg(exp, **{"fl.total_updates": 6, "eval.mc_dropout": "true", "model.mc_dropout_T": 3,
                           "eval.n_boot": 20})  # fmt: skip
    m = run_fl(tiny_scenario, cfg, 0, "unit_fin", resume=False)
    run_dir = next((tmp_path / "runs" / "unit_fin").iterdir())
    ref = {f: dict(np.load(run_dir / f)) for f in ("preds_val.npz", "preds_test.npz")}
    ref_summary = read_json(run_dir / "summary.json")
    for f in ("DONE", "metrics.json", "summary.json", "preds_val.npz", "preds_test.npz"):  # simulate a crash
        (run_dir / f).unlink()
    m2 = finalize_fl(run_dir, tiny_scenario, OmegaConf.load(run_dir / "config.yaml"))
    assert (run_dir / "DONE").exists()
    for f, arrs in ref.items():
        new = np.load(run_dir / f)
        for k, v in arrs.items():
            np.testing.assert_array_equal(new[k], v, err_msg=f"{f}:{k}")
    same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # NaN-safe  # noqa: E731
    assert same(m2["test"], m["test"]) and same(m2["val"], m["val"])
    for k in ("versions", "bytes_total", "best_val_auprc", "clients", "budgets"):
        assert same(m2["fl"][k], m["fl"][k]), k
    # events.jsonl stores simulated times rounded to 1e-4 s
    assert m2["fl"]["sim_time"] == pytest.approx(m["fl"]["sim_time"], abs=1e-4)
    strip = lambda evs: [{k: v for k, v in e.items() if k != "t"} for e in evs]  # noqa: E731
    assert same(strip(m2["fl"]["evals"]), strip(m["fl"]["evals"]))
    s = read_json(run_dir / "summary.json")
    assert s["finalized_from_checkpoint"] and "peak_gpu_mem_mib" not in s
    assert s["eps"] == ref_summary["eps"] and s["test_auprc"] == ref_summary["test_auprc"]
    with pytest.raises(FileExistsError):
        finalize_fl(run_dir, tiny_scenario, OmegaConf.load(run_dir / "config.yaml"))
