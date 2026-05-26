"""Cluster split: KMeans on a top-k set, per-cluster logists on the same features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from .modeling import make_logistic_baseline_pipeline


@dataclass(frozen=True, slots=True)
class ClusterSplitConfig:
    """KMeans + per-cluster models on one shared top-k subset."""

    feature_set: str = "top_10"
    n_clusters: int = 2
    min_cluster_samples: int = 80
    max_targets: int = 1000
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_clusters < 2:
            msg = f"n_clusters must be >= 2, got {self.n_clusters}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ClusterSplitTargetingResult:
    """OOF probabilities and cluster diagnostics."""

    oof_probabilities: np.ndarray
    features: tuple[str, ...]
    feature_count: int
    cluster_counts_train: dict[int, int]
    n_clusters: int


def fit_train_cluster_labels(
    X: pd.DataFrame,
    features: list[str],
    n_clusters: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """KMeans cluster ids on full train (scaled features)."""
    scaler = RobustScaler()
    matrix = _scale_cluster_matrix(X, features, scaler, fit=True)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(matrix)
    return kmeans.predict(matrix)


def build_cluster_y_profile(y: pd.Series, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster class counts and positive rate vs global baseline."""
    y_array = y.to_numpy()
    global_rate = float(y_array.mean())
    rows: list[dict[str, float | int]] = []
    for cluster_id in sorted(int(c) for c in np.unique(labels)):
        mask = labels == cluster_id
        n = int(mask.sum())
        n_positive = int(y_array[mask].sum())
        rate = n_positive / n if n else 0.0
        lift = rate / global_rate if n and global_rate > 0 else float("nan")
        rows.append({
            "cluster": cluster_id,
            "n": n,
            "n_positive": n_positive,
            "n_negative": n - n_positive,
            "positive_rate": rate,
            "lift_vs_global": lift,
        })
    profile = pd.DataFrame(rows)
    profile["global_positive_rate"] = global_rate
    return profile


def _scale_cluster_matrix(
    X: pd.DataFrame,
    features: list[str],
    scaler: RobustScaler,
    *,
    fit: bool = False,
) -> np.ndarray:
    matrix = X[features]
    if fit:
        return scaler.fit_transform(matrix)
    return scaler.transform(matrix)


def compute_cluster_split_oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    config: ClusterSplitConfig,
) -> ClusterSplitTargetingResult:
    """OOF probabilities: per-fold KMeans and cluster-specific logists."""
    y_array = y.to_numpy()
    n = len(y)
    oof = np.zeros(n, dtype=float)
    splitter = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    n_clusters = config.n_clusters

    for train_idx, val_idx in splitter.split(X, y):
        x_tr = X.iloc[train_idx]
        x_val = X.iloc[val_idx]
        y_tr = y_array[train_idx]

        seg_scaler = RobustScaler()
        seg_matrix_tr = _scale_cluster_matrix(x_tr, features, seg_scaler, fit=True)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=config.random_state,
            n_init=10,
        )
        kmeans.fit(seg_matrix_tr)
        seg_val = _scale_cluster_matrix(x_val, features, seg_scaler)
        clusters_tr = kmeans.predict(seg_matrix_tr)
        clusters_val = kmeans.predict(seg_val)

        fallback = make_logistic_baseline_pipeline(C=0.1)
        fallback.fit(x_tr[features], y_tr)

        val_probs = np.zeros(len(val_idx), dtype=float)
        for cluster_id in range(n_clusters):
            val_mask = clusters_val == cluster_id
            if not np.any(val_mask):
                continue
            train_mask = clusters_tr == cluster_id
            if int(train_mask.sum()) >= config.min_cluster_samples:
                model = make_logistic_baseline_pipeline(C=0.1)
                model.fit(x_tr.loc[train_mask, features], y_tr[train_mask])
                val_probs[val_mask] = model.predict_proba(x_val.loc[val_mask, features])[:, 1]
            else:
                val_probs[val_mask] = fallback.predict_proba(x_val.loc[val_mask, features])[:, 1]
        oof[val_idx] = val_probs

    seg_scaler_full = RobustScaler()
    full_seg = _scale_cluster_matrix(X, features, seg_scaler_full, fit=True)
    kmeans_full = KMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        n_init=10,
    )
    kmeans_full.fit(full_seg)
    labels_full = kmeans_full.predict(full_seg)
    cluster_counts = {int(c): int((labels_full == c).sum()) for c in range(n_clusters)}

    return ClusterSplitTargetingResult(
        oof_probabilities=oof,
        features=tuple(features),
        feature_count=len(features),
        cluster_counts_train=cluster_counts,
        n_clusters=n_clusters,
    )


def predict_cluster_split_test_probabilities(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    features: list[str],
    config: ClusterSplitConfig,
) -> np.ndarray:
    """Fit KMeans on train and score test with per-cluster models."""
    n_clusters = config.n_clusters
    seg_scaler = RobustScaler()
    seg_train = _scale_cluster_matrix(x_train, features, seg_scaler, fit=True)
    seg_test = _scale_cluster_matrix(x_test, features, seg_scaler)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        n_init=10,
    )
    kmeans.fit(seg_train)
    clusters_tr = kmeans.predict(seg_train)
    clusters_test = kmeans.predict(seg_test)

    fallback = make_logistic_baseline_pipeline(C=0.1)
    fallback.fit(x_train[features], y_train)

    test_probs = np.zeros(len(x_test), dtype=float)
    for cluster_id in range(n_clusters):
        test_mask = clusters_test == cluster_id
        if not np.any(test_mask):
            continue
        train_mask = clusters_tr == cluster_id
        if int(train_mask.sum()) >= config.min_cluster_samples:
            model = make_logistic_baseline_pipeline(C=0.1)
            model.fit(
                x_train.loc[train_mask, features],
                y_train.iloc[np.where(train_mask)[0]],
            )
            test_probs[test_mask] = model.predict_proba(x_test.loc[test_mask, features])[:, 1]
        else:
            test_probs[test_mask] = fallback.predict_proba(x_test.loc[test_mask, features])[:, 1]
    return test_probs
