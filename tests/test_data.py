"""Data pipeline tests (M1) on synthetic fixtures: splits, nodes, causality, normalisation, labels, subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedguard.config import DataConfig
from fedguard.data.download import DownloadError, RemoteFile, _is_valid, list_s3, select_subset, verify
from fedguard.data.features import DYN, causal_ffill_and_age, channel_spec
from fedguard.data.physionet2019 import (
    assign_clients,
    build_patient_table,
    list_raw_files,
    parse_all,
    process,
    unit_of,
)
from fedguard.data.windows import (
    NormStats,
    ProcessedData,
    WindowDataset,
    build_client_arrays,
    client_patient_indices,
)
from tests.fixtures_data import write_fixture

N = 40


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    frames = write_fixture(d, n_per_hospital=N)
    return d, frames


@pytest.fixture
def processed(raw, tmp_path):
    d, frames = raw
    cfg = DataConfig()
    out = tmp_path / "processed"
    manifest = process(cfg, d, out, workers=1)
    return ProcessedData.load(out), cfg, frames, manifest


# ---------------------------------------------------------------- download / verification
def test_verify_counts(raw):
    d, _ = raw
    assert verify(d, expected={"training_setA": N, "training_setB": N}) == {
        "training_setA": N,
        "training_setB": N,
    }
    with pytest.raises(DownloadError, match="expected 20336"):
        verify(d)  # real expected counts -> must fail loudly on the fixture
    (d / "training_setA" / "p000001.psv").unlink()
    with pytest.raises(DownloadError, match=f"found {N - 1}"):
        verify(d, expected={"training_setA": N, "training_setB": N})


def test_md5_and_size_validation(tmp_path):
    import hashlib

    f = tmp_path / "p1.psv"
    f.write_bytes(b"abc" * 100)
    good = RemoteFile("training_setA", "p1.psv", "u", 300, hashlib.md5(b"abc" * 100).hexdigest())
    assert _is_valid(f, good, check_md5=True)
    assert not _is_valid(f, RemoteFile("training_setA", "p1.psv", "u", 301, None), check_md5=True)
    assert not _is_valid(f, RemoteFile("training_setA", "p1.psv", "u", 300, "0" * 32), check_md5=True)


def test_list_s3_parses_pagination():
    page1 = b"""<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>true</IsTruncated><NextContinuationToken>tok</NextContinuationToken>
      <Contents><Key>challenge-2019/1.0.0/training/training_setA/p000001.psv</Key>
      <ETag>&quot;99565908185565399d1a3d399d2be53c&quot;</ETag><Size>9062</Size></Contents></ListBucketResult>"""
    page2 = b"""<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <Contents><Key>challenge-2019/1.0.0/training/training_setA/p000002.psv</Key>
      <ETag>&quot;abc-2&quot;</ETag><Size>10</Size></Contents></ListBucketResult>"""

    class Resp:
        def __init__(self, c):
            self.content, self.status_code = c, 200

    class Sess:
        def __init__(self):
            self.calls = []

        def get(self, url, timeout, params):
            self.calls.append(dict(params))
            return Resp(page1 if "continuation-token" not in params else page2)

    s = Sess()
    files = list_s3("training_setA", s)  # type: ignore[arg-type]
    assert [f.name for f in files] == ["p000001.psv", "p000002.psv"]
    assert files[0].md5 == "99565908185565399d1a3d399d2be53c" and files[0].size == 9062
    assert files[1].md5 is None  # multipart-style ETag is not an MD5
    assert s.calls[1]["continuation-token"] == "tok"


def test_fast_subset_is_deterministic(raw):
    names = [f"p{i:06d}.psv" for i in range(500)]
    a = select_subset(names, 50, seed=42)
    assert a == select_subset(list(reversed(names)), 50, seed=42)  # independent of listing order
    assert a != select_subset(names, 50, seed=43) and len(a) == 50
    d, _ = raw
    f1 = list_raw_files(d, 10, seed=42)
    assert f1 == list_raw_files(d, 10, seed=42) and len(f1) == 20


# ---------------------------------------------------------------- parsing, nodes, splits
def test_unit_assignment():
    nan = np.array([np.nan, np.nan])
    assert unit_of(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == ("MICU", False)
    assert unit_of(np.array([0.0, np.nan]), np.array([1.0, np.nan])) == ("SICU", False)
    assert unit_of(nan, nan) == ("UNK", False)
    assert unit_of(np.array([1.0]), np.array([1.0])) == ("UNK", True)


def test_patients_in_exactly_one_split_and_node(processed):
    data, cfg, frames, manifest = processed
    p = data.patients
    assert len(p) == len(frames) == 2 * N
    assert p["patient_id"].is_unique
    assert set(p["split"]) <= {0, 1, 2}
    splits = pd.read_csv(data.dir / "splits.csv", dtype={"patient_id": str})
    assert splits["patient_id"].is_unique and len(splits) == len(p)
    assert set(splits["split"]) <= {"train", "val", "test"}
    assert (p["stratum"] == p["hospital"] + "_" + p["unit"]).all()
    assert manifest["checks"]["unit_conflicts"] == 2  # one "BOTH" patient per hospital -> UNK
    # every included patient is in exactly one client for every partition
    for part, pol in [
        ("unit", "exclude"),
        ("unit", "separate"),
        ("hospital", "merge_into_hospital"),
        ("dirichlet", "separate"),
    ]:
        c = assign_clients(p, cfg.model_copy(update={"partition": part, "unk_policy": pol}))
        assert len(c) == len(p)
        if pol == "exclude":
            assert c[p["unit"] == "UNK"].isna().all() and c[p["unit"] != "UNK"].notna().all()
        else:
            assert c.notna().all()


def test_splits_are_order_independent_and_stratified(raw):
    d, _ = raw
    files = list_raw_files(d, None, 42)
    parsed = parse_all(files, workers=1)
    t1 = build_patient_table(parsed, (0.7, 0.15, 0.15), 42).set_index("patient_id")["split"]
    rev = list(reversed(parsed))
    t2 = build_patient_table(rev, (0.7, 0.15, 0.15), 42).set_index("patient_id")["split"]
    pd.testing.assert_series_equal(t1.sort_index(), t2.sort_index())
    t3 = build_patient_table(parsed, (0.7, 0.15, 0.15), 7).set_index("patient_id")["split"]
    assert not t1.sort_index().equals(t3.sort_index())


def test_parallel_parse_matches_inline(raw):
    d, _ = raw
    files = list_raw_files(d, None, 42)
    a, b = parse_all(files, workers=1), parse_all(files, workers=2)
    assert [p.patient_id for p in a] == [p.patient_id for p in b]
    assert [p.hospital for p in a] == [p.hospital for p in b]
    assert {p.hospital for p in a if p.patient_id.startswith("p1")} == {"B"}


def test_merge_into_hospital_invalid_for_unit_partition(processed):
    data, cfg, _, _ = processed
    with pytest.raises(ValueError, match="merge_into_hospital"):
        assign_clients(data.patients, cfg.model_copy(update={"unk_policy": "merge_into_hospital"}))


def test_labels_unchanged_from_raw(processed):
    data, _, frames, _ = processed
    for i, pid in enumerate(data.patients["patient_id"]):
        seg = np.asarray(data.label[data.offsets[i] : data.offsets[i + 1]])
        np.testing.assert_array_equal(seg, frames[pid]["SepsisLabel"].to_numpy())
        raw_vals = frames[pid][DYN].to_numpy(np.float32)
        np.testing.assert_array_equal(
            np.asarray(data.values_raw[data.offsets[i] : data.offsets[i + 1]]), raw_vals
        )
    row = data.patients[data.patients["ever_septic"] == 1].iloc[0]
    assert row["onset_hour"] == row["first_pos_row"] + 6


# ---------------------------------------------------------------- causality
def test_ffill_is_causal():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(30, 5)).astype(np.float32)
    v[rng.random(v.shape) > 0.3] = np.nan
    f, h = causal_ffill_and_age(v)
    for t in range(30):
        w = v.copy()
        w[t + 1 :] = rng.normal(size=w[t + 1 :].shape)  # rewrite the future
        f2, h2 = causal_ffill_and_age(w)
        np.testing.assert_array_equal(f[: t + 1], f2[: t + 1])
        np.testing.assert_array_equal(h[: t + 1], h2[: t + 1])
    # value before first measurement stays NaN (no backward fill)
    col = np.array([[np.nan], [np.nan], [3.0], [np.nan]], np.float32)
    f, h = causal_ffill_and_age(col)
    assert np.isnan(f[:2]).all() and f[2, 0] == 3.0 and f[3, 0] == 3.0 and h[3, 0] == 1.0


def test_windows_are_causal_end_to_end(raw, tmp_path):
    """Changing a *test* patient's raw rows after hour t (and reprocessing) leaves window t unchanged."""
    d, frames = raw
    cfg = DataConfig()
    out1 = tmp_path / "p1"
    process(cfg, d, out1, workers=1)
    data1 = ProcessedData.load(out1)
    clients = assign_clients(data1.patients, cfg.model_copy(update={"unk_policy": "separate"}))
    test_idx = client_patient_indices(data1.patients, clients, None, "test")
    victim = int(test_idx[np.argmax(data1.patients.loc[test_idx, "n_rows"].to_numpy())])
    pid = data1.patients.loc[victim, "patient_id"]
    t = int(data1.patients.loc[victim, "n_rows"]) // 2
    df = frames[pid].copy()
    df.loc[t + 1 :, DYN] = 999.0
    df.loc[t + 1 :, "SepsisLabel"] = 1
    sub = "training_setA" if pid < "p100000" else "training_setB"
    df.to_csv(d / sub / f"{pid}.psv", sep="|", index=False, na_rep="NaN")
    out2 = tmp_path / "p2"
    process(cfg, d, out2, workers=1)
    data2 = ProcessedData.load(out2)

    train = client_patient_indices(data1.patients, clients, None, "train")
    stats1, stats2 = NormStats.fit(data1, train), NormStats.fit(data2, train)
    np.testing.assert_array_equal(stats1.mean, stats2.mean)  # test patient cannot touch stats
    w1 = WindowDataset(build_client_arrays(data1, np.array([victim]), stats1, cfg), 24)
    w2 = WindowDataset(build_client_arrays(data2, np.array([victim]), stats2, cfg), 24)
    for hour in range(t + 1):
        for a, b in zip(w1[hour], w2[hour], strict=True):
            np.testing.assert_array_equal(a, b)
    assert not np.array_equal(w1[t + 1][0], w2[t + 1][0])  # sanity: the future change is visible later


def test_window_padding_and_patient_boundaries(processed):
    data, cfg, _, _ = processed
    idx = np.arange(len(data.patients))
    stats = NormStats.fit(data, idx)
    arr = build_client_arrays(data, idx, stats, cfg)
    ds = WindowDataset(arr, lookback=24)
    assert len(ds) == int(data.patients["n_rows"].sum())
    spec = channel_spec(True, True, list(cfg.features.static))
    x, m, y = ds[int(arr.offsets[1])]  # first hour of the 2nd patient
    assert x.shape == (24, spec.n_channels) and m.sum() == 1 and m[-1]
    assert np.all(x[:-1] == 0)  # no rows from the previous patient
    xb, mb, yb = ds.__getitems__([0, 1, 2, int(arr.offsets[1])])
    assert xb.shape == (4, 24, spec.n_channels) and mb.dtype == bool
    np.testing.assert_array_equal(xb[3], x)
    np.testing.assert_array_equal(yb, arr.label[[0, 1, 2, int(arr.offsets[1])]])


# ---------------------------------------------------------------- normalisation
def test_norm_stats_from_train_patients_only(processed):
    data, cfg, _, _ = processed
    clients = assign_clients(data.patients, cfg)
    for client in ["A_MICU", "B_SICU"]:
        tr = client_patient_indices(data.patients, clients, client, "train")
        stats = NormStats.fit(data, tr)
        v = np.concatenate([np.asarray(data.values_raw[data.offsets[i] : data.offsets[i + 1]]) for i in tr])
        with np.errstate(invalid="ignore"):
            exp = np.nanmean(v, 0)
        ok = ~np.isnan(exp)
        np.testing.assert_allclose(stats.mean[ok], exp[ok], rtol=1e-5)
        assert stats.n_patients == len(tr)
        # val/test patients of this client are not in the training index set
        others = np.concatenate(
            [client_patient_indices(data.patients, clients, client, s) for s in ("val", "test")]
        )
        assert not set(tr) & set(others)


def test_client_indices_respect_partition(processed):
    data, cfg, _, _ = processed
    clients = assign_clients(data.patients, cfg)
    pooled = client_patient_indices(data.patients, clients, None, "train")
    per = np.concatenate(
        [
            client_patient_indices(data.patients, clients, c, "train")
            for c in sorted(clients.dropna().unique())
        ]
    )
    assert sorted(pooled) == sorted(per)
    assert (data.patients.loc[pooled, "unit"] != "UNK").all()


def test_eda_runs(processed, tmp_path):
    from fedguard.data.eda import run_eda

    data, cfg, _, _ = processed
    s = run_eda(data, cfg, tmp_path / "eda")
    for f in [
        "summary_by_stratum.csv",
        "missingness.csv",
        "prevalence.png",
        "los.png",
        "missingness.png",
        "eda_summary.json",
    ]:
        assert (tmp_path / "eda" / f).exists()
    all_row = [r for r in s["main"] if r["node"] == "ALL"][0]
    assert all_row["patients"] == int((data.patients["unit"] != "UNK").sum())
