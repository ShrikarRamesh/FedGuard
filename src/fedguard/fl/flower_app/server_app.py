"""Flower ServerApp: FedAvg / FedProx with Flower's built-in strategies (Message API, Flower 1.38).

The server holds no patient data. Every round it saves the global model and appends events in the same
``events.jsonl`` format as the in-house engine (time = real wall-clock seconds), so the Streamlit app can
show a live deployment. Validation metrics come from the clients (each on its own validation set) and are
logged as a weighted mean; the exact global metrics for the cross-check are computed afterwards from the
saved checkpoints with ``fedguard fl eval-checkpoints`` (D19).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from fedguard.data.features import channel_spec
from fedguard.fl.flower_app.common import cfg_from_run_config
from fedguard.train import loops
from fedguard.utils.io import append_jsonl, runs_dir, write_json
from fedguard.utils.seed import seed_everything

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = dict(context.run_config)
    cfg = cfg_from_run_config(rc)
    seed_everything(cfg.seed)
    out = Path(str(rc.get("out-dir") or "")) if rc.get("out-dir") else None
    if out is None:
        out = runs_dir() / "flower" / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.seed}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    events = out / "events.jsonl"
    t0 = time.time()
    n_clients = int(rc.get("num-clients", 4))

    def log(type_: str, **kw) -> None:
        append_jsonl(
            events, {"t": round(time.time() - t0, 3), "wall": round(time.time() - t0, 3), "type": type_, **kw}
        )

    # identical initialisation to the in-house engine: seed_everything(seed); manual_seed(seed); build
    torch.manual_seed(cfg.seed)
    f = cfg.data.features
    n_ch = channel_spec(f.use_masks, f.use_deltas, list(f.static)).n_channels
    model = loops.build_model(cfg.model, n_ch, int(cfg.data.lookback))
    init = ArrayRecord(model.state_dict())
    from omegaconf import OmegaConf

    write_json(out / "config.json", {"run_config": rc, "cfg": OmegaConf.to_container(cfg, resolve=True)})
    log("config", mode="sync", algorithm=f"flower_{cfg.fl.algorithm}", clients=None, runtime="flower-deployment",
        n_clients=n_clients, model_bytes=sum(v.numel() * v.element_size() for v in model.state_dict().values()),
        dp=False)  # fmt: skip

    def train_aggr(records: list[RecordDict], key: str) -> MetricRecord:
        for r in records:
            m = r["metrics"]
            log("update_received", client_index=int(m["client-index"]), samples=int(m.get("samples", 0)),
                loss=float(m["train_loss"]), staleness=0)  # fmt: skip
        tot = sum(float(r["metrics"][key]) for r in records)
        return MetricRecord(
            {
                "train_loss": sum(
                    float(r["metrics"]["train_loss"]) * float(r["metrics"][key]) for r in records
                )
                / tot
            }
        )

    def eval_aggr(records: list[RecordDict], key: str) -> MetricRecord:
        out_m: dict[str, float] = {}
        for k in ("val_auroc", "val_auprc"):
            rs = [r["metrics"] for r in records if k in r["metrics"]]
            if rs:
                out_m[k] = sum(float(m[k]) * float(m[key]) for m in rs) / sum(float(m[key]) for m in rs)
        log("eval", val_metric="client-weighted-mean", **out_m)
        return MetricRecord(out_m)

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        torch.save(
            {"model": arrays.to_torch_state_dict()}, out / "checkpoints" / f"round_{server_round:03d}.pt"
        )
        log("aggregate", round=server_round, version=server_round)
        return None

    common = dict(fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=n_clients, min_evaluate_nodes=n_clients,
                  min_available_nodes=n_clients, train_metrics_aggr_fn=train_aggr, evaluate_metrics_aggr_fn=eval_aggr)  # fmt: skip
    if cfg.fl.algorithm == "fedprox":
        strategy = FedProx(proximal_mu=float(cfg.fl.prox_mu), **common)
    else:
        strategy = FedAvg(**common)
    result = strategy.start(grid=grid, initial_arrays=init, num_rounds=int(cfg.fl.rounds), train_config=ConfigRecord({}),
                            evaluate_fn=evaluate_fn)  # fmt: skip
    torch.save({"model": result.arrays.to_torch_state_dict()}, out / "checkpoints" / "final.pt")
    log("done", version=int(cfg.fl.rounds))
