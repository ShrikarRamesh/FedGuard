"""Paths and small file helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Repository root: the nearest ancestor of the CWD containing ``pyproject.toml``.

    Falls back to the package's own location (for an editable install) and finally the CWD.
    """
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "pyproject.toml").exists() and (base / "src" / "fedguard").exists():
            return base
    here = Path(__file__).resolve()
    for base in here.parents:
        if (base / "pyproject.toml").exists():
            return base
    return Path.cwd()


def data_dir() -> Path:
    """Root data directory (``FEDGUARD_DATA_DIR`` or ``<repo>/data``)."""
    return Path(os.environ.get("FEDGUARD_DATA_DIR", repo_root() / "data"))


def runs_dir() -> Path:
    """Root directory for run outputs (``FEDGUARD_RUNS_DIR`` or ``<repo>/runs``)."""
    return Path(os.environ.get("FEDGUARD_RUNS_DIR", repo_root() / "runs"))


def results_dir() -> Path:
    """Aggregated results directory (``<repo>/results``)."""
    return repo_root() / "results"


def configs_dir() -> Path:
    """Config directory (``<repo>/configs``)."""
    return repo_root() / "configs"


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Write JSON atomically (write to a temp file, then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record per line (used for events.jsonl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
