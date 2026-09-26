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
def data_download(fast: Fast = False) -> None:
    """Download PhysioNet/CinC 2019 training data and verify file counts."""
    _not_yet("M1", "fedguard data download")


@data_app.command("process")
def data_process(fast: Fast = False, seed: Seed = 42, override: Overrides = None) -> None:
    """Parse raw .psv files, assign nodes, make patient-level splits, write processed arrays."""
    _not_yet("M1", "fedguard data process")


@data_app.command("eda")
def data_eda(fast: Fast = False) -> None:
    """Per-node summary tables and plots into results/eda/."""
    _not_yet("M1", "fedguard data eda")


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
