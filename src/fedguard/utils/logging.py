"""Logging setup: console + per-run log file, plus a CSV/JSON metrics logger that always works offline."""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(
    name: str = "fedguard", log_file: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Return a logger writing to stderr and optionally to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr for h in logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target = str(log_file.resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    logger.propagate = False
    return logger


class MetricsLogger:
    """Append-only metrics log: ``metrics.csv`` (wide) in the run dir; optional W&B mirror.

    W&B is used only if ``use_wandb`` and ``WANDB_API_KEY`` is set in the environment.
    """

    def __init__(self, run_dir: Path, use_wandb: bool = False, wandb_kwargs: dict[str, Any] | None = None):
        self.path = Path(run_dir) / "metrics.csv"
        self._fields: list[str] | None = None
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                header = f.readline().strip()
                self._fields = header.split(",") if header else None
        self._wandb = None
        if use_wandb:
            if not os.environ.get("WANDB_API_KEY"):
                get_logger().warning("--wandb given but WANDB_API_KEY is not set; logging to CSV only.")
            else:
                import wandb  # optional extra

                self._wandb = wandb.init(**(wandb_kwargs or {}))

    def log(self, row: dict[str, Any]) -> None:
        """Log one row of scalar metrics. New keys after the first row are rejected loudly."""
        if self._fields is None:
            self._fields = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        unknown = set(row) - set(self._fields)
        if unknown:
            raise KeyError(
                f"metrics keys {sorted(unknown)} not in header {self._fields}; log them in a separate file"
            )
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(row)
        if self._wandb is not None:
            self._wandb.log(row)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
