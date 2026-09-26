# FedGuard

Privacy-preserving federated learning for real-time ICU sepsis early warning (B.E. CSE AI & ML final-year project).

> Work in progress. Status and all real numbers so far: [PROGRESS.md](PROGRESS.md). Design decisions: [docs/decisions.md](docs/decisions.md). No results have been produced yet.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate         # Linux
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # CUDA build (use /cpu for CPU-only)
pip install -e ".[dev]"
make test
```

Windows (PowerShell). Keep the venv outside OneDrive:

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\fedguard
C:\Users\<you>\.venvs\fedguard\Scripts\Activate.ps1
.\scripts\make.ps1 install
.\scripts\make.ps1 test
```

`FEDGUARD_DATA_DIR` and `FEDGUARD_RUNS_DIR` relocate `data/` and `runs/` (e.g. off a synced folder).

## Commands

`fedguard --help` lists everything. `fedguard info` prints versions and resolved paths. Every command accepts `--fast` for a tiny smoke-test run.

The full reproduce-everything guide and demo runbook are written in M10.
