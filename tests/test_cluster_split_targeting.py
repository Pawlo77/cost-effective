"""Tests for two-cluster split targeting."""

import numpy as np
import pandas as pd

from cost_effective.models.cluster_split_targeting import (
    ClusterSplitConfig,
    build_cluster_y_profile,
    compute_cluster_split_oof_probabilities,
    fit_train_cluster_labels,
    predict_cluster_split_test_probabilities,
)


def _toy_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(1)
    cols = [f"var_{i}" for i in range(10)]
    x = pd.DataFrame(rng.normal(size=(n, 10)), columns=cols)
    y = pd.Series((x["var_0"] + x["var_1"] > 0).astype(int))
    x_test = pd.DataFrame(rng.normal(size=(50, 10)), columns=cols)
    return x, y, x_test


def test_cluster_split_oof_shape() -> None:
    x, y, _ = _toy_data()
    config = ClusterSplitConfig(n_clusters=2, min_cluster_samples=20, cv_folds=3)
    features = list(x.columns[:6])
    result = compute_cluster_split_oof_probabilities(x, y, features, config)
    assert len(result.oof_probabilities) == len(y)
    assert 0.0 <= result.oof_probabilities.min() <= result.oof_probabilities.max() <= 1.0
    assert result.feature_count == 6
    assert result.features == tuple(features)
    assert len(result.cluster_counts_train) == 2


def test_cluster_y_profile_rates() -> None:
    x, y, _ = _toy_data()
    labels = fit_train_cluster_labels(x, list(x.columns[:3]), n_clusters=2, random_state=0)
    profile = build_cluster_y_profile(y, labels)
    assert profile["n"].sum() == len(y)
    assert int(profile["n_positive"].sum()) == int(y.sum())
    assert len(profile) == 2


def test_cluster_split_test_probabilities_shape() -> None:
    x, y, x_test = _toy_data()
    config = ClusterSplitConfig(n_clusters=2, min_cluster_samples=20, cv_folds=3)
    features = list(x.columns[:6])
    probs = predict_cluster_split_test_probabilities(x, y, x_test, features, config)
    assert probs.shape == (len(x_test),)
