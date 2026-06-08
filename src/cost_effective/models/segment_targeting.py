"""Segment-aware ranking: cluster customers, per-segment models, global top-k."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from .modeling import make_logistic_baseline_pipeline


@dataclass(frozen=True, slots=True)
class SegmentConfig:
    """Hyperparameters for GMM segmentation + per-cluster logistic."""

    n_clusters: int = 5
    segment_feature_set: str = "top_10"
    model_feature_set: str = "top_03"
    min_cluster_samples: int = 80
    max_targets: int = 1000
    cv_folds: int = 5
    random_state: int = 42
    pca_components: int = 5
    use_pca: bool = True


@dataclass(frozen=True, slots=True)
class SegmentTargetingResult:
    """OOF probabilities and cluster diagnostics."""

    oof_probabilities: np.ndarray
    segment_features: tuple[str, ...]
    model_features: tuple[str, ...]
    feature_count: int
    cluster_counts_train: dict[int, int]
    n_clusters: int


def _fit_segment_transform(
    X: pd.DataFrame,
    segment_features: list[str],
    config: SegmentConfig,
) -> tuple[RobustScaler, PCA | None]:
    """Fit scaler (and optional PCA) on segment features."""
    scaler = RobustScaler()
    scaler.fit(X[segment_features])
    pca: PCA | None = None
    if config.use_pca:
        n_cols = len(segment_features)
        n_comp = min(config.pca_components, n_cols)
        if n_comp < n_cols:
            scaled = scaler.transform(X[segment_features])
            pca = PCA(n_components=n_comp, random_state=config.random_state)
            pca.fit(scaled)
    return scaler, pca


def _transform_segment_matrix(
    X: pd.DataFrame,
    segment_features: list[str],
    scaler: RobustScaler,
    pca: PCA | None,
) -> np.ndarray:
    scaled = scaler.transform(X[segment_features])
    if pca is not None:
        return pca.transform(scaled)
    return scaled


def compute_segment_oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    segment_features: list[str],
    model_features: list[str],
    config: SegmentConfig,
) -> SegmentTargetingResult:
    """OOF positive probabilities with per-fold GMM and cluster logists."""
    y_array = y.to_numpy()
    n = len(y)
    oof = np.zeros(n, dtype=float)
    splitter = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    for train_idx, val_idx in splitter.split(X, y):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y_array[train_idx]

        seg_scaler = RobustScaler()
        seg_matrix_tr = seg_scaler.fit_transform(X_tr[segment_features])
        pca = None
        if config.use_pca and seg_matrix_tr.shape[1] > config.pca_components:
            pca = PCA(n_components=config.pca_components, random_state=config.random_state)
            seg_matrix_tr = pca.fit_transform(seg_matrix_tr)

        gmm = GaussianMixture(
            n_components=config.n_clusters,
            random_state=config.random_state,
            n_init=3,
        )
        gmm.fit(seg_matrix_tr)
        seg_val = _transform_segment_matrix(X_val, segment_features, seg_scaler, pca)
        clusters_tr = gmm.predict(seg_matrix_tr)
        clusters_val = gmm.predict(seg_val)

        fallback = make_logistic_baseline_pipeline(C=0.1)
        fallback.fit(X_tr[model_features], y_tr)

        val_probs = np.zeros(len(val_idx), dtype=float)
        for cluster_id in range(config.n_clusters):
            val_mask = clusters_val == cluster_id
            if not np.any(val_mask):
                continue
            train_mask = clusters_tr == cluster_id
            if int(train_mask.sum()) >= config.min_cluster_samples:
                model = make_logistic_baseline_pipeline(C=0.1)
                model.fit(X_tr.loc[train_mask, model_features], y_tr[train_mask])
                val_probs[val_mask] = model.predict_proba(X_val.loc[val_mask, model_features])[:, 1]
            else:
                val_probs[val_mask] = fallback.predict_proba(X_val.loc[val_mask, model_features])[
                    :, 1
                ]
        oof[val_idx] = val_probs

    seg_scaler_full, pca_full = _fit_segment_transform(X, segment_features, config)
    full_seg = _transform_segment_matrix(X, segment_features, seg_scaler_full, pca_full)
    gmm_full = GaussianMixture(
        n_components=config.n_clusters,
        random_state=config.random_state,
        n_init=3,
    )
    gmm_full.fit(full_seg)
    labels_full = gmm_full.predict(full_seg)
    cluster_counts = {int(c): int((labels_full == c).sum()) for c in range(config.n_clusters)}

    return SegmentTargetingResult(
        oof_probabilities=oof,
        segment_features=tuple(segment_features),
        model_features=tuple(model_features),
        feature_count=len(model_features),
        cluster_counts_train=cluster_counts,
        n_clusters=config.n_clusters,
    )


def predict_segment_test_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    segment_features: list[str],
    model_features: list[str],
    config: SegmentConfig,
) -> np.ndarray:
    """Fit segmentation on train and score test with per-cluster models."""
    seg_scaler, pca = _fit_segment_transform(X_train, segment_features, config)
    seg_train = _transform_segment_matrix(X_train, segment_features, seg_scaler, pca)
    seg_test = _transform_segment_matrix(X_test, segment_features, seg_scaler, pca)
    gmm = GaussianMixture(
        n_components=config.n_clusters,
        random_state=config.random_state,
        n_init=3,
    )
    gmm.fit(seg_train)
    clusters_tr = gmm.predict(seg_train)
    clusters_test = gmm.predict(seg_test)
    fallback = make_logistic_baseline_pipeline(C=0.1)
    fallback.fit(X_train[model_features], y_train)

    test_probs = np.zeros(len(X_test), dtype=float)
    for cluster_id in range(config.n_clusters):
        test_mask = clusters_test == cluster_id
        if not np.any(test_mask):
            continue
        train_mask = clusters_tr == cluster_id
        if int(train_mask.sum()) >= config.min_cluster_samples:
            model = make_logistic_baseline_pipeline(C=0.1)
            model.fit(
                X_train.loc[train_mask, model_features],
                y_train.iloc[np.where(train_mask)[0]],
            )
            test_probs[test_mask] = model.predict_proba(X_test.loc[test_mask, model_features])[:, 1]
        else:
            test_probs[test_mask] = fallback.predict_proba(X_test.loc[test_mask, model_features])[
                :, 1
            ]
    return test_probs
