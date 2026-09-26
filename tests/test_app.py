"""App smoke tests (Streamlit AppTest): every page renders without exceptions, with artefacts present
(synthetic ones written to a temp results dir) and with them missing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture(autouse=True)
def _restore_main():
    """AppTest runs each page as ``__main__`` and leaves it installed; spawned worker processes of later tests
    (e.g. the parallel parser) would then re-import a Streamlit page and crash. Restore the real ``__main__``.
    """
    import sys

    main, argv = sys.modules.get("__main__"), list(sys.argv)
    yield
    if main is not None:
        sys.modules["__main__"] = main
    sys.argv[:] = argv


PAGES = [APP / "streamlit_app.py", *sorted((APP / "pages").glob("*.py"))]


def _write_artifacts(root: Path) -> None:
    import sys

    sys.path.insert(0, str(APP))
    import data  # app/data.py demo generators (used only to produce well-formed test artefacts)

    root.mkdir(parents=True, exist_ok=True)
    (root / "attack").mkdir(exist_ok=True)
    (root / "results.json").write_text(json.dumps(data._demo_results()))
    (root / "results_full.json").write_text(json.dumps(data._demo_full()))
    (root / "attack" / "attack.json").write_text(json.dumps(data._demo_attack()))
    (root / "tables").mkdir(exist_ok=True)
    (root / "tables" / "main.md").write_text("| method | auroc |\n|---|---|\n| x | 0.5 |\n")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_on_pipeline_artifacts(page, monkeypatch):
    """Used by `make smoke`: render every page on the artefacts the --fast pipeline just produced."""
    import os

    res, runs = os.environ.get("FEDGUARD_SMOKE_RESULTS_DIR"), os.environ.get("FEDGUARD_SMOKE_RUNS_DIR")
    if not res:
        pytest.skip("only in `make smoke` (FEDGUARD_SMOKE_RESULTS_DIR unset)")
    monkeypatch.setenv("FEDGUARD_RESULTS_DIR", res)
    if runs:
        monkeypatch.setenv("FEDGUARD_RUNS_DIR", runs)
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_train_together_replays_events(tmp_path, monkeypatch):
    """Page 1 renders a recorded run (events.jsonl in the engine's format)."""
    run = tmp_path / "runs" / "fedguard" / "20260101-000000_0"
    run.mkdir(parents=True)
    (run / "DONE").write_text("x")
    ev = [
        {"t": 0, "type": "config", "mode": "async", "clients": ["A_MICU", "B_SICU"], "dp": True,
         "n_samples": {"A_MICU": 10, "B_SICU": 20},
         "budgets": {c: {"target_eps": 3.0, "delta": 1e-5} for c in ("A_MICU", "B_SICU")}},
        {"t": 0, "type": "eval", "version": 0, "val_auroc": 0.5, "eps": {"A_MICU": 0, "B_SICU": 0}},
        {"t": 1.5, "type": "dispatch", "client": "A_MICU", "version": 0},
        {"t": 3.0, "type": "merge", "client": "A_MICU", "version": 1, "staleness": 0, "weight": 0.3, "eps": {"A_MICU": 0.4, "B_SICU": 0}},
        {"t": 3.0, "type": "eval", "version": 1, "val_auroc": 0.6, "eps": {"A_MICU": 0.4, "B_SICU": 0}},
    ]  # fmt: skip
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    monkeypatch.setenv("FEDGUARD_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FEDGUARD_RESULTS_DIR", str(tmp_path / "results"))
    at = AppTest.from_file(str(APP / "pages" / "1_Train_together.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == "fedguard · seed 0"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
@pytest.mark.parametrize("present", [True, False], ids=["artifacts", "missing"])
def test_page_renders(page, present, tmp_path, monkeypatch):
    root = tmp_path / "results"
    if present:
        _write_artifacts(root)
    else:
        root.mkdir()
    monkeypatch.setenv("FEDGUARD_RESULTS_DIR", str(root))
    monkeypatch.delenv("FEDGUARD_DEMO", raising=False)
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    if not present:
        text = " ".join(w.value for w in at.warning) + " ".join(i.value for i in at.info)
        assert (
            "Run `fedguard" in text
            or "No Flower" in text
            or "fedguard fl run" in text
            or page.name == "streamlit_app.py"
            or text
        )
