"""FedGuard command-line interface: ``fedguard <group> <command> [--fast] [--config ...] [--seed N] [-o key=value]``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from fedguard.config import load_config

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="FedGuard: federated, private ICU sepsis early warning."
)
data_app = typer.Typer(no_args_is_help=True, help="Download, process and explore PhysioNet 2019 data.")
train_app = typer.Typer(no_args_is_help=True, help="Centralized and local-only training.")
fl_app = typer.Typer(no_args_is_help=True, help="Federated learning (in-house engine and Flower).")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(fl_app, name="fl")

# Shared option types
Fast = Annotated[bool, typer.Option("--fast", help="Tiny patient subset, 1-2 epochs/rounds (smoke test).")]
Seed = Annotated[int, typer.Option("--seed", help="Random seed.")]
Overrides = Annotated[
    list[str] | None, typer.Option("--override", "-o", help="Config override, e.g. -o train.lr=3e-4")
]
ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Experiment config YAML.")]
Wandb = Annotated[
    bool, typer.Option("--wandb", help="Mirror metrics to Weights & Biases (needs WANDB_API_KEY).")
]


def _not_yet(milestone: str, what: str) -> None:
    typer.secho(
        f"`{what}` is not implemented yet (planned in {milestone}; see PROGRESS.md).", fg="yellow", err=True
    )
    raise typer.Exit(code=2)


@app.command()
def info() -> None:
    """Print environment, library versions and resolved data/run paths."""
    from fedguard.utils.io import data_dir, repo_root, runs_dir
    from fedguard.utils.runs import git_info, library_versions

    out = {
        "repo_root": str(repo_root()),
        "data_dir": str(data_dir()),
        "runs_dir": str(runs_dir()),
        "git": git_info(),
        "versions": library_versions(),
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("show-config")
def show_config(
    paths: Annotated[list[str], typer.Argument(help="Config files, e.g. experiments/centralized")],
    fast: Fast = False,
    override: Overrides = None,
) -> None:
    """Print the composed, resolved config (useful to check --fast and overrides)."""
    from omegaconf import OmegaConf

    typer.echo(OmegaConf.to_yaml(load_config(*paths, overrides=override, fast=fast), resolve=True))


# ---------------------------------------------------------------- data (M1)
@data_app.command("download")
def data_download(
    fast: Fast = False,
    source: Annotated[str, typer.Option(help="s3 (MD5-verified) or physionet (fallback)")] = "s3",
    override: Overrides = None,
) -> None:
    """Download PhysioNet/CinC 2019 training data (resumable, MD5-checked) and verify file counts."""
    from fedguard.config import load_data_config
    from fedguard.data.download import download, verify
    from fedguard.utils.io import data_dir

    dcfg = load_data_config(fast=fast, overrides=override)
    raw = data_dir() / dcfg.raw_subdir
    typer.echo(f"Downloading to {raw} ({'fast subset' if fast else 'full'}) ...")
    stats = download(
        raw,
        subset_per_hospital=dcfg.subset_per_hospital,
        seed=dcfg.split_seed,
        source=source,
        progress=lambda s, i, n: typer.echo(f"  {s}: {i}/{n}"),
    )
    typer.echo(f"Transfer: {stats}")
    expected = None if fast else dcfg.expected_counts
    counts = verify(raw, expected=expected)
    if fast:  # the fast subset must contain exactly the requested files
        for s, c in counts.items():
            if c < (dcfg.subset_per_hospital or 0):
                raise typer.BadParameter(f"{s}: only {c} files present after fast download")
    typer.secho(f"Verified: {counts}", fg="green")


def processed_dir(fast: bool, dcfg) -> Path:
    """Processed data dir; ``--fast`` uses a separate ``<processed>_fast`` dir so it never clobbers real data."""
    from fedguard.utils.io import data_dir

    return data_dir() / (dcfg.processed_subdir + ("_fast" if fast else ""))


@data_app.command("process")
def data_process(
    fast: Fast = False,
    workers: Annotated[
        int | None, typer.Option(help="Parser processes (default: CPUs - 1; 1 = inline)")
    ] = None,
    override: Overrides = None,
) -> None:
    """Parse raw .psv files, assign strata, make patient-level splits, write processed arrays."""
    from fedguard.config import load_data_config
    from fedguard.data.physionet2019 import process
    from fedguard.utils.io import data_dir

    dcfg = load_data_config(fast=fast, overrides=override)
    out = processed_dir(fast, dcfg)
    typer.echo(f"Processing {data_dir() / dcfg.raw_subdir} -> {out}")
    manifest = process(
        dcfg, data_dir() / dcfg.raw_subdir, out, workers=workers,
        progress=lambda i, n: typer.echo(f"  parsed chunk {i}/{n}") if i % 20 == 0 or i == n else None,
    )  # fmt: skip
    typer.secho(json.dumps(manifest, indent=2), fg="green")


@data_app.command("export-node")
def data_export_node(
    node: Annotated[str, typer.Option(help="Node to export, e.g. A_MICU")],
    out: Annotated[Path, typer.Option(help="Output dir to copy to that hospital's laptop")],
    fast: Fast = False, override: Overrides = None,
) -> None:  # fmt: skip
    """Write a processed-data dir with ONLY one node's patients (for a Flower SuperNode on its own laptop)."""
    from fedguard.config import load_data_config
    from fedguard.data.physionet2019 import assign_clients, export_node
    from fedguard.data.windows import ProcessedData

    dcfg = load_data_config(fast=fast, overrides=override)
    src = processed_dir(fast, dcfg)
    clients = assign_clients(ProcessedData.load(src).patients, dcfg)
    if node not in set(clients.dropna()):
        raise typer.BadParameter(f"unknown node {node!r}; available: {sorted(clients.dropna().unique())}")
    m = export_node(src, out, (clients == node).to_numpy())
    typer.secho(f"{node}: {m['n_patients']} patients, {m['n_rows']} rows -> {out}", fg="green")


@data_app.command("eda")
def data_eda(fast: Fast = False, override: Overrides = None) -> None:
    """Per-node summary tables and plots into results/eda/ (results/fast/eda/ with --fast)."""
    from fedguard.config import load_data_config
    from fedguard.data.eda import run_eda
    from fedguard.data.windows import ProcessedData
    from fedguard.utils.io import out_root

    dcfg = load_data_config(fast=fast, overrides=override)
    out = out_root(fast) / "eda"
    summary = run_eda(ProcessedData.load(processed_dir(fast, dcfg)), dcfg, out)
    cols = ["node", "patients", "septic_patients", "septic_patient_rate", "rows", "positive_row_rate"]
    import pandas as pd

    typer.echo(pd.DataFrame(summary["main"])[cols].to_string(index=False))
    typer.secho(f"EDA written to {out}", fg="green")


# ---------------------------------------------------------------- training (M2-M3)
ModelOpt = Annotated[str, typer.Option("--model", "-m", help="patchtst | gru | lr | lgbm")]
SeedsOpt = Annotated[
    str | None, typer.Option("--seeds", help="Comma-separated seeds (overrides --seed), e.g. 0,1,2")
]
NameOpt = Annotated[
    str | None, typer.Option("--name", help="Experiment name (default: config's `experiment`)")
]


def _seeds(seed: int, seeds: str | None) -> list[int]:
    return [int(s) for s in seeds.split(",")] if seeds else [seed]


def _scenario(cfg, fast: bool):
    """Load the processed data + partition described by ``cfg.data``."""
    from fedguard.config import validate_data
    from fedguard.data.scenario import Scenario

    dcfg = validate_data(cfg)
    return Scenario.load(processed_dir(fast, dcfg), dcfg)


def _exp_name(base: str, model: str, fast: bool) -> str:
    return f"{base}_{model}" + ("_fast" if fast else "")


@train_app.command("centralized")
def train_centralized(
    config: ConfigOpt = None, model: ModelOpt = "patchtst", fast: Fast = False, seed: Seed = 0, seeds: SeedsOpt = None,
    override: Overrides = None, wandb: Wandb = False, name: NameOpt = None,
) -> None:  # fmt: skip
    """Train on pooled data from all nodes (upper bound)."""
    from fedguard.config import experiment_config
    from fedguard.train.centralized import run_centralized

    cfg = experiment_config(config or "experiments/centralized", fast, override, {"model": f"model/{model}"})
    cfg.wandb = wandb
    sc = _scenario(cfg, fast)
    for s in _seeds(seed, seeds):
        res = run_centralized(sc, cfg, s, _exp_name(name or cfg.experiment, model, fast))
        _echo_result(res)


@train_app.command("local")
def train_local(
    node: Annotated[str | None, typer.Option(help="Node to train on, e.g. A_MICU")] = None,
    all_nodes: Annotated[bool, typer.Option("--all-nodes")] = False,
    config: ConfigOpt = None, model: ModelOpt = "patchtst", fast: Fast = False, seed: Seed = 0,
    seeds: SeedsOpt = None, override: Overrides = None, wandb: Wandb = False, name: NameOpt = None,
) -> None:  # fmt: skip
    """Train each node alone on its own data (lower bound)."""
    from fedguard.config import experiment_config
    from fedguard.train.local import run_local

    if not node and not all_nodes:
        raise typer.BadParameter("give --node NAME or --all-nodes")
    cfg = experiment_config(config or "experiments/local", fast, override, {"model": f"model/{model}"})
    cfg.wandb = wandb
    sc = _scenario(cfg, fast)
    nodes = sc.client_names if all_nodes else [node]
    for s in _seeds(seed, seeds):
        for n in nodes:
            _echo_result(run_local(sc, cfg, s, n, _exp_name(name or cfg.experiment, model, fast)))


def _echo_result(res) -> None:
    if isinstance(res, Path):
        typer.echo(f"skipped (already done): {res}")
        return
    g = res["test"]["global"]
    typer.secho(
        f"test AUROC {g['auroc']:.4f}  AUPRC {g['auprc']:.4f} (prevalence {g['prevalence']:.4f})  "
        f"ECE {g['ece']:.4f}  [{res['wall_clock_s']:.0f} s]",
        fg="green",
    )


# ---------------------------------------------------------------- federated (M4-M5)
@fl_app.command("run")
def fl_run(
    config: ConfigOpt = None, fast: Fast = False, seed: Seed = 0, seeds: SeedsOpt = None,
    override: Overrides = None, wandb: Wandb = False,
    name: Annotated[str | None, typer.Option(help="Experiment name (default: config's `experiment`)")] = None,
) -> None:  # fmt: skip
    """Run an in-house FL simulation (sync FedAvg/FedProx or async FedGuard, optionally with DP)."""
    from fedguard.config import experiment_config
    from fedguard.fl.runner import run_fl

    cfg = experiment_config(config or "experiments/fedavg", fast, override)
    cfg.wandb = wandb
    sc = _scenario(cfg, fast)
    exp = (name or cfg.experiment) + ("_fast" if fast else "")
    for s in _seeds(seed, seeds):
        _echo_result(run_fl(sc, cfg, s, exp))


@fl_app.command("eval-checkpoints")
def fl_eval_checkpoints(
    run_dir: Annotated[Path, typer.Argument(help="Flower run dir (contains config.json + checkpoints/)")],
    fast: Fast = False,
) -> None:
    """Score a Flower deployment run's saved round checkpoints with the in-house evaluation (cross-check)."""
    from omegaconf import OmegaConf

    from fedguard.fl.runner import eval_checkpoints
    from fedguard.utils.io import read_json

    cfg = OmegaConf.create(read_json(run_dir / "config.json")["cfg"])
    res = eval_checkpoints(run_dir, _scenario(cfg, bool(cfg.get("fast", fast))), cfg,
                           n_boot=int(cfg.get("eval", {}).get("n_boot", 1000)))  # fmt: skip
    g = res["test"]["global"]
    typer.secho(
        f"best {res['best_round_ckpt']}: test AUROC {g['auroc']:.4f} AUPRC {g['auprc']:.4f}", fg="green"
    )


# ---------------------------------------------------------------- later milestones
ExpOpt = Annotated[
    str, typer.Option("--experiment", "-e", help="Experiment dir name under runs/, e.g. fedguard")
]


def _run_dirs(experiment: str, fast: bool, seeds: str | None) -> list[Path]:
    from fedguard.utils.runs import completed_runs

    exp = experiment + ("_fast" if fast and not experiment.endswith("_fast") else "")
    runs = completed_runs(exp)
    if not runs:
        raise typer.BadParameter(f"no completed runs for experiment {exp!r}; train it first")
    keep = _seeds(0, seeds) if seeds else sorted(runs)
    return [runs[s] for s in keep if s in runs]


@app.command()
def alerts(experiment: ExpOpt = "fedguard", fast: Fast = False, seeds: SeedsOpt = None,
           refractory: Annotated[int, typer.Option(help="Refractory hours")] = 6) -> None:  # fmt: skip
    """Tune alert thresholds on validation, evaluate alert policies + calibration on test (M6)."""
    from fedguard.alerts.run import run_alerts
    from fedguard.eval.run_artifacts import run_config, run_scenario

    for d in _run_dirs(experiment, fast, seeds):
        sc = run_scenario(run_config(d))
        r = run_alerts(d, sc.data.patients, refractory=refractory)
        t, g = r["test"]["threshold_only"], r["test"]["fedguard"]
        typer.secho(
            f"{d.name}: tau_r={r['params']['tau_r']:.3f} tau_s={r['params']['tau_s']:.4f} | threshold-only "
            f"{t['false_per_100h']:.2f} FA/100h sens {t['sensitivity']:.3f} | gated {g['false_per_100h']:.2f} "
            f"FA/100h sens {g['sensitivity']:.3f}",
            fg="green",
        )


@app.command("mc-ablation")
def mc_ablation(experiment: ExpOpt = "fedguard", fast: Fast = False, seeds: SeedsOpt = "0") -> None:
    """MC-Dropout T in {5,10,20,50}: ECE and alert metrics (thresholds re-tuned on validation per T)."""
    from fedguard.alerts.mc_ablation import run_mc_ablation

    for d in _run_dirs(experiment, fast, seeds):
        r = run_mc_ablation(d, Ts=(2, 5) if fast else (5, 10, 20, 50))
        for row in r["rows"]:
            typer.echo(f"T={row['T']:3d} ECE={row['ece_mc']:.4f} gated FA/100h={row['fedguard']['false_per_100h']:.2f} "
                       f"sens={row['fedguard']['sensitivity']:.3f}")  # fmt: skip


@app.command()
def explain(experiment: ExpOpt = "fedguard", fast: Fast = False, seeds: SeedsOpt = None,
            n_windows: Annotated[int, typer.Option(help="Test windows for global IG")] = 1000) -> None:  # fmt: skip
    """Global Integrated-Gradients feature importance + attention rollout for finished runs (M6)."""
    from fedguard.explain.run import run_explain

    for d in _run_dirs(experiment, fast, seeds):
        r = run_explain(d, n_windows=200 if fast else n_windows)
        typer.secho(f"{d.name}: top variables {r['ranking'][:8]}", fg="green")


@app.command()
def attack(
    experiment: ExpOpt = "fedavg", fast: Fast = False, seed: Seed = 0,
    n: Annotated[int, typer.Option(help="Test windows (patients) to attack")] = 50,
    iters: Annotated[int, typer.Option(help="Optimisation steps per restart")] = 400,
    restarts: Annotated[int, typer.Option(help="Random restarts per window")] = 3,
) -> None:  # fmt: skip
    """Gradient-inversion attack on single-window updates, without DP and with FedGuard's DP (M7)."""
    from fedguard.attack.run import run_attack
    from fedguard.config import experiment_config
    from fedguard.utils.io import out_root

    victim = _run_dirs(experiment, fast, str(seed))[0]
    pcfg = experiment_config("experiments/fedguard", fast).privacy
    out = out_root(fast) / "attack"
    r = run_attack(victim, out, pcfg, n=4 if fast else n, iters=20 if fast else iters,
                   restarts=1 if fast else restarts, seed=seed)  # fmt: skip
    for name, c in r["conditions"].items():
        typer.echo(f"{name:10s} sigma={c['sigma']:.3f}  mean r={c['mean_r']:.3f} {c['mean_r_ci95']}  "
                   f"MSE={c['mse']:.3f}  label acc={c['label_accuracy']:.2f}")  # fmt: skip
    typer.secho(f"written to {out}", fg="green")


@app.command()
def report(fast: Fast = False) -> None:
    """Aggregate runs into results/summary.csv, tables (md + tex) and figures (M8)."""
    from fedguard.eval.report import run_report

    r = run_report(fast)
    typer.echo(
        r["tables"]["main"][["method", "n_seeds", "auroc", "auprc", "prevalence"]].to_string(index=False)
    )
    typer.secho(f"{r['summary']['n_runs']} runs aggregated; gap recovered (AUROC): {r['summary']['gap_recovered_auroc']}",
                fg="green")  # fmt: skip


@app.command()
def export(fast: Fast = False) -> None:
    """Write results/results.json (+ results_full.json) for the HTML and Streamlit front ends (M9)."""
    from fedguard.export.results_json import export as do_export
    from fedguard.export.results_json import validate

    r = do_export(fast)
    validate(r["results"])
    if r["missing"]:
        typer.secho(f"not run yet (written as null): {r['missing']}", fg="yellow")
    typer.secho(
        f"results.json valid, written to {r['path']} ({len(r['results']['patients'])} patients)", fg="green"
    )


if __name__ == "__main__":
    app()
