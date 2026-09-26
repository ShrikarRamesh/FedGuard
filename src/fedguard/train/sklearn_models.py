"""Logistic regression and LightGBM on hand-crafted window features (sanity floor baselines)."""

from __future__ import annotations

import numpy as np

from fedguard.data.windows import WindowDataset
from fedguard.models.baselines import handcrafted_features


def window_features(ds: WindowDataset, idx: np.ndarray | None = None, batch: int = 8192) -> np.ndarray:
    """Hand-crafted features for samples ``idx`` (default: all) of ``ds``, computed in batches."""
    idx = np.arange(len(ds)) if idx is None else np.asarray(idx)
    out = []
    for s in range(0, len(idx), batch):
        x, m, _ = ds.gather(idx[s : s + batch])
        out.append(handcrafted_features(x, m, ds.a.spec))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def fit_predict(
    kind: str, train: WindowDataset, val: WindowDataset, tests: list[WindowDataset], seed: int,
    max_train: int = 300_000,
) -> tuple[list[np.ndarray], np.ndarray, dict]:  # fmt: skip
    """Fit ``lr`` or ``lgbm``; return (test probabilities per test set, val probabilities, info).

    Training uses a seeded uniform subsample of at most ``max_train`` windows (memory); class imbalance is
    handled by class weighting (LR) / scale_pos_weight (LightGBM) from the training labels.
    """
    rng = np.random.default_rng(seed)
    tr_idx = np.arange(len(train))
    if len(tr_idx) > max_train:
        tr_idx = np.sort(rng.choice(tr_idx, max_train, replace=False))
    Xtr, ytr = window_features(train, tr_idx), train.labels()[tr_idx].astype(int)
    Xva, yva = window_features(val), val.labels().astype(int)
    info = {"n_train_windows": int(len(tr_idx)), "n_features": int(Xtr.shape[1])}
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)
        )
        clf.fit(Xtr, ytr)

        def prob(X):
            return clf.predict_proba(X)[:, 1]

    elif kind == "lgbm":
        import lightgbm as lgb

        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        clf = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, scale_pos_weight=spw, random_state=seed, verbose=-1, n_jobs=-1,
        )  # fmt: skip
        has_both = 0 < yva.sum() < len(yva)
        clf.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)] if has_both else None,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)] if has_both else None,
        )  # fmt: skip
        info["best_iteration"] = int(clf.best_iteration_ or clf.n_estimators)

        def prob(X):
            return clf.predict_proba(X)[:, 1]

    else:
        raise ValueError(f"unknown sklearn model {kind!r}")
    return [prob(window_features(t)) for t in tests], prob(Xva), info
