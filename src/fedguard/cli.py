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


@data_app.command("eda")
def data_eda(fast: Fast = False, override: Overrides = None) -> None:
    """Per-node summary tables and plots into results/eda/ (results/eda_fast/ with --fast)."""
    from fedguard.config import load_data_config
    from fedguard.data.eda import run_eda
    from fedguard.data.windows import ProcessedData
    from fedguard.utils.io import results_dir

    dcfg = load_data_config(fast=fast, overrides=override)
    out = results_dir() / ("eda_fast" if fast else "eda")
    summary = run_eda(ProcessedData.load(processed_dir(fast, dcfg)), dcfg, out)
    cols = ["node", "patients", "septic_patients", "septic_patient_rate", "rows", "positive_row_rate"]
    import pandas as pd

    typer.echo(pd.DataFrame(summary["main"])[cols].to_string(index=False))
    typer.secho(f"EDA written to {out}", fg="green")


# ---------------------------------------------------------------- training (M2-M3)
@train_app.command("centralized")
def train_centralized(
    config: ConfigOpt = None,
    fast: Fast = False,
    seed: Seed = 0,
    override: Overrides = None,
    wandb: Wandb = False,
) -> None:
    """Train on pooled data from all nodes (upper bound)."""
    _not_yet("M2/M3", "fedguard train centralized")


@train_app.command("local")
def train_local(
    node: Annotated[str | None, typer.Option(help="Node to train on, e.g. A_MICU")] = None,
    all_nodes: Annotated[bool, typer.Option("--all-nodes")] = False,
    config: ConfigOpt = None, fast: Fast = False, seed: Seed = 0, override: Overrides = None, wandb: Wandb = False,
) -> None:  # fmt: skip
    """Train each node alone on its own data (lower bound)."""
    _not_yet("M3", "fedguard train local")


# ---------------------------------------------------------------- federated (M4-M5)
@fl_app.command("run")
def fl_run(
    config: ConfigOpt = None,
    fast: Fast = False,
    seed: Seed = 0,
    override: Overrides = None,
    wandb: Wandb = False,
) -> None:
    """Run an in-house FL simulation (sync FedAvg/FedProx or async FedGuard, optionally with DP)."""
    _not_yet("M4", "fedguard fl run")


# ---------------------------------------------------------------- later milestones
@app.command()
def alerts(config: ConfigOpt = None, fast: Fast = False) -> None:
    """Tune alert thresholds on validation, evaluate alert policies on test (M6)."""
    _not_yet("M6", "fedguard alerts")


@app.command()
def attack(config: ConfigOpt = None, fast: Fast = False, seed: Seed = 0) -> None:
    """Gradient-inversion attack with and without DP (M7)."""
    _not_yet("M7", "fedguard attack")


@app.command()
def report(fast: Fast = False) -> None:
    """Aggregate runs into results/summary.csv, tables and figures (M8)."""
    _not_yet("M8", "fedguard report")


@app.command()
def export(fast: Fast = False) -> None:
    """Write results/results.json for the HTML and Streamlit front ends (M9)."""
    _not_yet("M9", "fedguard export")


if __name__ == "__main__":
    app()
