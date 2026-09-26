"""results.json contract tests: schema from section 11.1 of the build prompt + equal-length rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from fedguard.export.results_json import METHOD_NAMES, RESULTS_SCHEMA, validate


def example() -> dict:
    H = 5
    return {
        "methods": [{"name": n, "auroc": 0.8, "auprc": 0.1} for n in METHOD_NAMES],
        "prevalence": 0.016,
        "privacy": {"eps": [1, 2, 3, 5, 8], "uniform": [0.6] * 5, "adaptive": [0.61, None, 0.63, 0.64, 0.65], "no_dp": 0.7},
        "alerts": {"threshold_only": {"false_per_100h": 3.0, "sensitivity": 0.8}, "fedguard": {"false_per_100h": 2.5, "sensitivity": 0.79}},
        "patients": [{
            "name": "Test patient p000001", "onset_hour": 3, "hr": [80.0] * H, "map": [None, 70.0, 71, 72, 73],
            "resp": [16] * H, "temp": [37.0] * H, "spo2": [97] * H, "risk_mean": [0.1] * H, "risk_std": [0.02] * H,
            "attr_features": ["HR", "MAP"], "attr": [[0.1, -0.2]] * H,
        }],
    }  # fmt: skip


def test_example_validates():
    validate(example())


def test_method_names_match_html_contract():
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "fedguard_demo.html").read_text(
        encoding="utf-8"
    )
    for n in METHOD_NAMES:
        assert f"name:'{n}'" in html, n


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["patients"][0].update(hr=[1.0]),  # length mismatch
        lambda r: r["patients"][0].update(attr=[[0.1]] * 5),  # attr width != attr_features
        lambda r: r["methods"].append({"name": "Mystery", "auroc": 0.9, "auprc": 0.1}),  # unknown method
        lambda r: r.pop("alerts"),  # missing key
        lambda r: r["patients"][0].update(risk_mean=[1.5] * 5),  # probability out of range
    ],
)
def test_invalid_results_rejected(mutate):
    r = copy.deepcopy(example())
    mutate(r)
    with pytest.raises((jsonschema.ValidationError, ValueError, AssertionError)):
        validate(r)


def test_exported_file_if_present_is_valid():
    """If the pipeline has produced results.json (real, or the --fast one during `make smoke`), it must satisfy
    the contract."""
    import os

    from fedguard.utils.io import repo_root

    root = os.environ.get("FEDGUARD_SMOKE_RESULTS_DIR") or str(repo_root() / "results")
    p = Path(root) / "results.json"
    if not p.exists():
        pytest.skip("results/results.json not produced yet (run `fedguard export`)")
    validate(json.loads(p.read_text(encoding="utf-8")))
    jsonschema.validate(json.loads(p.read_text(encoding="utf-8")), RESULTS_SCHEMA)
