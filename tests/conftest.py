"""Shared fixtures. Tests never touch the real data/ or runs/ directories."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Point FEDGUARD_DATA_DIR / FEDGUARD_RUNS_DIR at a per-test temp dir."""
    monkeypatch.setenv("FEDGUARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FEDGUARD_RUNS_DIR", str(tmp_path / "runs"))
    yield
