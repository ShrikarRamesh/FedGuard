# Common commands. Windows without make: use `scripts\make.ps1 <target>` (same targets).
PY ?= python

.PHONY: install test test-all lint format smoke app clean-runs

install:
	$(PY) -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

test-all:
	$(PY) -m pytest -q -m ""

lint:
	$(PY) -m ruff check src tests app
	$(PY) -m black --check src tests app

format:
	$(PY) -m ruff check --fix src tests app
	$(PY) -m black src tests app

# End-to-end --fast pipeline. Steps are added as milestones land (see PROGRESS.md).
smoke:
	$(PY) -m fedguard.cli info

app:
	$(PY) -m streamlit run app/streamlit_app.py
