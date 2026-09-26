"""Flower ClientApp: one hospital. Trains only on its own node's patients (node config ``client``)."""

from __future__ import annotations

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedguard.data.windows import WindowDataset
from fedguard.eval import metrics as M
from fedguard.fl.client import FLClient
from fedguard.fl.flower_app.common import cfg_from_run_config, scenario_for
from fedguard.train import loops

app = ClientApp()
_CACHE: dict[str, object] = {}


def _client(context: Context) -> tuple[FLClient, object]:
    """Build (once per process) the FLClient exactly as the in-house engine does (same seed formula)."""
    key = f"{context.run_id}:{context.node_config['client']}"
    if key not in _CACHE:
        cfg = cfg_from_run_config(dict(context.run_config))
        sc = scenario_for(cfg, dict(context.node_config))
        name = str(context.node_config["client"])
        if name not in sc.client_names:
            raise ValueError(f"node {name!r} not in the local data ({sc.client_names})")
        idx = sc.client_names.index(name)
        device = loops.get_device(cfg.fl.get("device", "auto"))
        torch.manual_seed(cfg.seed)
        f = sc.cfg.features
        from fedguard.data.features import channel_spec

        n_ch = channel_spec(f.use_masks, f.use_deltas, list(f.static)).n_channels
        model = loops.build_model(cfg.model, n_ch, sc.cfg.lookback).to(device)
        arr = sc.arrays(name, "train", "site")
        cl = FLClient(
            name=name, train_arrays=arr, lookback=sc.cfg.lookback, model=model, device=device,
            seed=cfg.seed * 101 + idx, lr=float(cfg.fl.lr), weight_decay=float(cfg.fl.get("weight_decay", 0.01)),
            batch_size=int(cfg.fl.batch_size), local_steps=int(cfg.fl.local_steps),
            grad_clip=cfg.fl.get("grad_clip", 1.0), pos_weight=loops.pos_weight_from(arr.label),
            prox_mu=float(cfg.fl.get("prox_mu", 0.0)),
        )  # fmt: skip
        val = WindowDataset(sc.arrays(name, "val", "site"), sc.cfg.lookback)
        _CACHE[key] = (cl, val, idx)
    return _CACHE[key][0], _CACHE[key]  # type: ignore[index,return-value]


@app.train()
def train(msg: Message, context: Context) -> Message:
    cl, (_, _, idx) = _client(context)
    mu = msg.content["config"].get("proximal-mu", None)
    if mu is not None:
        cl.prox_mu = float(mu)
    # With subprocess isolation every message runs in a fresh process, so the participation counter must
    # come from the server (FedAvg puts "server-round" in the config); local_train increments it.
    cl._round = int(msg.content["config"]["server-round"]) - 1
    state = msg.content["arrays"].to_torch_state_dict()
    res = cl.local_train(state)
    metrics = MetricRecord({"num-examples": res["n"], "train_loss": float(res["loss"]), "client-index": idx,
                            "samples": int(res["samples"])})  # fmt: skip
    return Message(RecordDict({"arrays": ArrayRecord(res["state"]), "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on this node's own validation set (for the live dashboard)."""
    cl, (_, val, idx) = _client(context)
    model = cl.core()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    p = loops.predict(model, val, cl.device)
    y = val.labels().astype(np.float64)
    s = M.summary(y, p)
    rec = {"num-examples": len(y), "client-index": idx}
    for k in ("auroc", "auprc"):
        if np.isfinite(s[k]):
            rec[f"val_{k}"] = float(s[k])
    return Message(RecordDict({"metrics": MetricRecord(rec)}), reply_to=msg)
