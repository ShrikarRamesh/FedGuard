"""Run directories: ``runs/<experiment>/<timestamp>_<seed>/`` with resolved config, provenance and a DONE marker."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fedguard.utils.io import read_json, repo_root, runs_dir, write_json

TRACKED_PACKAGES = (
    "torch", "opacus", "flwr", "captum", "streamlit", "lightgbm", "scikit-learn",
    "numpy", "pandas", "pyarrow", "omegaconf", "pydantic",
)  # fmt: skip

DONE_MARKER = "DONE"


def git_info() -> dict[str, Any]:
    """Current git commit hash and whether the working tree is dirty (``None`` if not a git repo)."""
    root = repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}


def library_versions() -> dict[str, str | None]:
    """Installed versions of the tracked packages, plus Python and CUDA info."""
    out: dict[str, str | None] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        import torch

        out["cuda_available"] = str(torch.cuda.is_available())
        out["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["torch_cuda"] = torch.version.cuda
    except ImportError:
        pass
    return out


def config_hash(cfg: DictConfig | dict[str, Any]) -> str:
    """Short SHA-256 of the resolved config (keys sorted), used to match runs for resumable sweeps."""
    conf = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    container = OmegaConf.to_container(conf, resolve=True)
    return hashlib.sha256(json.dumps(container, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Run:
    """A single run directory."""

    dir: Path
    experiment: str
    seed: int

    @property
    def done(self) -> bool:
        return (self.dir / DONE_MARKER).exists()

    def mark_done(self, summary: dict[str, Any] | None = None) -> None:
        """Mark the run complete; optionally write a final ``summary.json``."""
        if summary is not None:
            write_json(self.dir / "summary.json", summary)
        (self.dir / DONE_MARKER).write_text(datetime.now().isoformat(), encoding="utf-8")

    @property
    def checkpoints(self) -> Path:
        p = self.dir / "checkpoints"
        p.mkdir(exist_ok=True)
        return p


def create_run(
    experiment: str, seed: int, cfg: DictConfig | dict[str, Any], argv: list[str] | None = None,
    root: Path | None = None,
) -> Run:  # fmt: skip
    """Create ``<root>/<experiment>/<timestamp>_<seed>/`` and write config + provenance into it."""
    root = Path(root) if root is not None else runs_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / experiment / f"{stamp}_{seed}"
    i = 1
    while run_dir.exists():  # two runs in the same second
        run_dir = root / experiment / f"{stamp}-{i}_{seed}"
        i += 1
    run_dir.mkdir(parents=True)
    conf = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    OmegaConf.save(conf, run_dir / "config.yaml", resolve=True)
    write_json(
        run_dir / "meta.json",
        {
            "experiment": experiment,
            "seed": seed,
            "config_hash": config_hash(conf),
            "created": datetime.now().isoformat(),
            "argv": argv if argv is not None else sys.argv,
            "git": git_info(),
            "versions": library_versions(),
        },
    )
    return Run(dir=run_dir, experiment=experiment, seed=seed)


def find_completed(
    experiment: str, seed: int, config_hash: str | None = None, root: Path | None = None
) -> Path | None:
    """Return a completed run dir for (experiment, seed[, config hash]) if one exists; used for resumable sweeps."""
    base = (Path(root) if root is not None else runs_dir()) / experiment
    if not base.exists():
        return None
    for d in sorted(base.iterdir(), reverse=True):
        if not (d.is_dir() and d.name.endswith(f"_{seed}") and (d / DONE_MARKER).exists()):
            continue
        if config_hash is None:
            return d
        meta_hash = read_json(d / "meta.json").get("config_hash") if (d / "meta.json").exists() else None
        if meta_hash == config_hash:
            return d
    return None
