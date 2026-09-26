"""Write results/results.json (hard contract with app/static/fedguard_demo.html) and results_full.json.

Every number comes from a finished run's artefacts; a value that has not been produced yet is written as
null (never a placeholder number), and ``results_full.json["missing"]`` lists what is missing.

Demo patients (docs/demo.md): drawn from the global TEST set with a fixed seed, by a rule that does not
look at model performance for septic patients:
  * 3 septic: seeded random among septic test patients with 36-96 h stays and an unambiguous onset;
  * 1 non-septic "false alarm avoided": seeded random among non-septic test patients (36-96 h) where
    threshold-only alerting raises >= 1 alarm and the uncertainty-gated policy raises none (spec example);
  * 2 non-septic: seeded random among the remaining non-septic test patients with 36-96 h stays.
Risk curves are the MC-Dropout mean/std (T = 50) of the `alerts_from` experiment's seed-0 model (default: async
federated learning without DP, D30); thresholds are the ones tuned on validation for that run. Vitals are raw clinical units, forward-filled within the stay (null before the
first measurement). Attributions are Integrated Gradients of the logit per variable, one row per hour
(the window ending at that hour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedguard.alerts.metrics import episode_starts
from fedguard.alerts.policy import threshold_only, uncertainty_gated
from fedguard.data.features import DYN
from fedguard.data.windows import WindowDataset, build_client_arrays
from fedguard.eval.report import collect, main_table, privacy_table
from fedguard.eval.run_artifacts import load_model, load_preds, run_config, run_norm, run_scenario
from fedguard.explain.integrated_gradients import integrated_gradients, per_variable
from fedguard.train import loops
from fedguard.utils.io import out_root, read_json, write_json
from fedguard.utils.runs import completed_runs

METHOD_NAMES = ["Local-only", "FedAvg", "FedProx", "FedAvg + DP", "FedGuard", "Centralized"]
EPS = [1, 2, 3, 5, 8]
VITAL_KEYS = {"hr": "HR", "map": "MAP", "resp": "Resp", "temp": "Temp", "spo2": "O2Sat"}
DISPLAY_ATTR = ["HR", "MAP", "O2Sat", "Resp", "Temp"]  # plus the top-3 labs by global importance


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _select_patients(preds, patients, params: dict, seed: int = 0) -> list[tuple[int, str]]:
    rng = np.random.default_rng(seed)
    offs = preds.offsets()
    pids = preds.patient_idx[offs[:-1]]
    los = np.diff(offs)
    ok_len = (los >= 36) & (los <= 96)
    septic = patients.loc[pids, "ever_septic"].to_numpy() == 1
    amb = patients.loc[pids, "onset_ambiguous"].to_numpy() == 1
    mean, std = preds.p_mc_mean, preds.p_mc_std
    tr, ts = params["tau_r"], params["tau_s"]

    def n_alarms(k: int, gated: bool) -> int:
        s, e = offs[k], offs[k + 1]
        a = uncertainty_gated(mean[s:e], std[s:e], tr, ts) if gated else threshold_only(mean[s:e], tr)
        return len(episode_starts(a, int(params["refractory"])))

    cand_sep = np.flatnonzero(septic & ok_len & ~amb)
    chosen = [(int(k), "septic") for k in rng.choice(cand_sep, min(3, len(cand_sep)), replace=False)]
    cand_non = np.flatnonzero(~septic & ok_len)
    avoided = [k for k in cand_non if n_alarms(k, False) > 0 and n_alarms(k, True) == 0]
    if avoided:
        chosen.append((int(rng.choice(avoided)), "false alarm avoided by the uncertainty gate"))
    rest = np.setdiff1d(cand_non, [c for c, _ in chosen])
    chosen += [(int(k), "non-septic") for k in rng.choice(rest, min(2, len(rest)), replace=False)]
    return chosen


def build_patients(run_dir: Path, top_labs: list[str], seed: int = 0) -> tuple[list[dict], dict]:
    cfg = run_config(run_dir)
    sc = run_scenario(cfg)
    norm = run_norm(run_dir)
    preds = load_preds(run_dir, "test")
    alerts = read_json(run_dir / "alerts.json")
    params = alerts["params"]
    chosen = _select_patients(preds, sc.data.patients, params, seed)
    device = loops.get_device("auto")
    from fedguard.data.windows import NormStats

    offs = preds.offsets()
    stats = NormStats.public() if norm == "public" else None
    model = None
    feats = DISPLAY_ATTR + [f for f in top_labs if f not in DISPLAY_ATTR][:3]
    out = []
    for k, why in chosen:
        pid = int(preds.patient_idx[offs[k]])
        s, e = offs[k], offs[k + 1]
        row = sc.data.patients.loc[pid]
        r0, r1 = int(sc.data.offsets[pid]), int(sc.data.offsets[pid + 1])
        raw = np.asarray(sc.data.values_ffill[r0:r1])
        p = {"name": f"Test patient {row['patient_id']}", "desc": None,
             "onset_hour": int(row["onset_hour"]) if row["ever_septic"] == 1 else None,
             "selection": why, "node": str(sc.clients.iloc[pid])}  # fmt: skip
        for key, var in VITAL_KEYS.items():
            p[key] = [_num(v) for v in raw[:, DYN.index(var)]]
        p["risk_mean"] = [float(v) for v in preds.p_mc_mean[s:e]]
        p["risk_std"] = [float(v) for v in preds.p_mc_std[s:e]]
        # attributions: IG for the window ending at every hour of the stay
        st = (
            stats
            if stats is not None
            else (sc.stats(sc.clients.iloc[pid]) if norm == "site" else sc.stats(None))
        )
        arr = build_client_arrays(sc.data, np.array([pid]), st, sc.cfg)
        ds = WindowDataset(arr, sc.cfg.lookback)
        if model is None:
            model = load_model(run_dir, cfg, arr.spec.n_channels, device)
        x, m, _ = ds.gather(np.arange(len(ds)))
        attr = (
            integrated_gradients(
                model, torch.from_numpy(x).to(device), torch.from_numpy(m).to(device), n_steps=32
            )
            .cpu()
            .numpy()
        )
        names, pv = per_variable(attr, arr.spec)
        per_hour = pv.sum(1)  # [hours, G]
        p["attr_features"] = ["SpO₂" if f == "O2Sat" else f for f in feats]
        p["attr"] = [[float(per_hour[h, names.index(f)]) for f in feats] for h in range(len(ds))]
        out.append(p)
    return out, {"run": str(run_dir), "tau_r": params["tau_r"], "tau_s": params["tau_s"], "refractory": params["refractory"],
                 "selection_seed": seed}  # fmt: skip


ALERT_SOURCE_LABELS = {
    "async_nodp": "async federated learning (FedGuard aggregation), no DP, MC-Dropout T=50",
    "fedguard": "FedGuard (async + adaptive DP, eps=3), MC-Dropout T=50",
    "centralized_patchtst": "centralized (pooled data, not permitted), MC-Dropout T=50",
}


def _alerts_of(experiment: str, fast: bool) -> list[dict]:
    runs = completed_runs(experiment + ("_fast" if fast else ""))
    return [read_json(d / "alerts.json") for d in runs.values() if (d / "alerts.json").exists()]


def export(fast: bool = False, alerts_from: str = "async_nodp") -> dict[str, Any]:
    """``alerts_from``: experiment whose alert metrics and seed-0 model drive results.json alerts + patients (D30)."""
    df = collect(fast)
    mt = main_table(df).set_index("method")
    pt = privacy_table(df)
    missing: list[str] = []
    methods = []
    for name in METHOD_NAMES:
        au = _num(mt.loc[name, "auroc_mean"]) if name in mt.index else None
        ap = _num(mt.loc[name, "auprc_mean"]) if name in mt.index else None
        if au is None:
            missing.append(f"method {name}")
        methods.append({"name": name, "auroc": au, "auprc": ap})
    cen = df[df.experiment == "centralized_patchtst"]
    prevalence = _num(cen.prevalence.mean()) if len(cen) else None
    eps_rows = pt[pt.epsilon.apply(lambda e: isinstance(e, float))].set_index("epsilon")
    lines = {}
    for key, label in (("uniform", "FedGuard (uniform rule)"), ("adaptive", "FedGuard (adaptive)")):
        vals = [_num(eps_rows.loc[float(e), f"_{label}_auroc"]) for e in EPS]
        missing += [f"privacy {key} eps={e}" for e, v in zip(EPS, vals, strict=True) if v is None]
        lines[key] = vals
    nodp = (
        _num(pt["_nodp_auroc"].dropna().iloc[0])
        if "_nodp_auroc" in pt and pt["_nodp_auroc"].notna().any()
        else None
    )
    fg_runs = completed_runs(alerts_from + ("_fast" if fast else ""))
    al = _alerts_of(alerts_from, fast)
    alerts = {}
    for pol, key in (("threshold_only", "threshold_only"), ("fedguard", "fedguard")):
        if al:
            alerts[key] = {"false_per_100h": _num(np.mean([a["test"][pol]["false_per_100h"] for a in al])),
                           "sensitivity": _num(np.mean([a["test"][pol]["sensitivity"] for a in al]))}  # fmt: skip
        else:
            alerts[key] = {"false_per_100h": None, "sensitivity": None}
            missing.append(f"alerts {key}")
    patients, pmeta = [], {}
    if 0 in fg_runs and (fg_runs[0] / "alerts.json").exists():
        ex = fg_runs[0] / "explain.json"
        top = (
            [v for v in read_json(ex)["ranking"] if v in DYN and v not in DISPLAY_ATTR][:3]
            if ex.exists()
            else ["Lactate", "WBC", "Creatinine"]
        )
        patients, pmeta = build_patients(fg_runs[0], top)
    else:
        missing.append(f"patients ({alerts_from} seed 0 with alerts.json)")
    res = {"methods": methods, "prevalence": prevalence,
           "privacy": {"eps": EPS, "uniform": lines["uniform"], "adaptive": lines["adaptive"], "no_dp": nodp},
           "alerts": alerts, "patients": patients}  # fmt: skip
    out_dir = out_root(fast)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", res)
    full = {
        "results": res,
        "missing": missing,
        "patients_meta": {
            **pmeta,
            "source": alerts_from,
            "source_label": ALERT_SOURCE_LABELS.get(alerts_from, alerts_from),
        },
        "alerts_source": alerts_from,
        "alerts_source_label": ALERT_SOURCE_LABELS.get(alerts_from, alerts_from),
        "alerts_by_model": {k: _alerts_of(k, fast) for k in ALERT_SOURCE_LABELS},
        "main_table": main_table(df).drop(columns=[], errors="ignore").to_dict(orient="records"),
        "privacy_table": pt.to_dict(orient="records"),
        "alerts_per_seed": al,
        "per_run": df.to_dict(orient="records"),
    }
    write_json(out_dir / "results_full.json", full)
    return {"results": res, "missing": missing, "path": str(out_dir / "results.json")}


RESULTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["methods", "prevalence", "privacy", "alerts", "patients"],
    "properties": {
        "methods": {"type": "array", "items": {"type": "object", "required": ["name", "auroc", "auprc"],
                    "properties": {"name": {"enum": METHOD_NAMES}, "auroc": {"type": ["number", "null"]},
                                   "auprc": {"type": ["number", "null"]}}}},  # fmt: skip
        "prevalence": {"type": ["number", "null"]},
        "privacy": {"type": "object", "required": ["eps", "uniform", "adaptive", "no_dp"],
                    "properties": {"eps": {"type": "array", "items": {"type": "number"}},
                                   "uniform": {"type": "array", "items": {"type": ["number", "null"]}},
                                   "adaptive": {"type": "array", "items": {"type": ["number", "null"]}},
                                   "no_dp": {"type": ["number", "null"]}}},  # fmt: skip
        "alerts": {"type": "object", "required": ["threshold_only", "fedguard"],
                   "properties": {k: {"type": "object", "required": ["false_per_100h", "sensitivity"]}
                                  for k in ("threshold_only", "fedguard")}},  # fmt: skip
        "patients": {"type": "array", "items": {
            "type": "object",
            "required": ["name", "onset_hour", "hr", "map", "resp", "temp", "spo2", "risk_mean", "risk_std", "attr_features", "attr"],
            "properties": {"name": {"type": "string"}, "onset_hour": {"type": ["integer", "null"]},
                           **{k: {"type": "array", "items": {"type": ["number", "null"]}} for k in ("hr", "map", "resp", "temp", "spo2")},
                           "risk_mean": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}},
                           "risk_std": {"type": "array", "items": {"type": "number", "minimum": 0}},
                           "attr_features": {"type": "array", "items": {"type": "string"}},
                           "attr": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}}},  # fmt: skip
    },
}


def validate(res: dict[str, Any]) -> None:
    """jsonschema validation + the equal-length rules the HTML relies on."""
    import jsonschema

    jsonschema.validate(res, RESULTS_SCHEMA)
    assert len(res["privacy"]["uniform"]) == len(res["privacy"]["adaptive"]) == len(res["privacy"]["eps"])
    for p in res["patients"]:
        n = len(p["risk_mean"])
        for k in ("hr", "map", "resp", "temp", "spo2", "risk_std", "attr"):
            if len(p[k]) != n:
                raise ValueError(f"{p['name']}: {k} has {len(p[k])} entries, risk_mean has {n}")
        if any(len(r) != len(p["attr_features"]) for r in p["attr"]):
            raise ValueError(f"{p['name']}: attr rows must have one column per attr_features entry")
