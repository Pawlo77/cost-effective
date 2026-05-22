"""Tests for segment-aware targeting."""

import numpy as np
import pandas as pd

from cost_effective.models.segment_targeting import (
    SegmentConfig,
    compute_segment_oof_probabilities,
    predict_segment_test_probabilities,
)


def _toy_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(1)
    cols = [f"var_{i}" for i in range(10)]
    x = pd.DataFrame(rng.normal(size=(n, 10)), columns=cols)
    y = pd.Series((x["var_0"] + x["var_1"] > 0).astype(int))
    x_test = pd.DataFrame(rng.normal(size=(50, 10)), columns=cols)
    return x, y, x_test


def test_segment_oof_shape() -> None:
    x, y, _ = _toy_data()
    config = SegmentConfig(
        n_clusters=3,
        min_cluster_samples=20,
        cv_folds=3,
        use_pca=True,
        pca_components=3,
    )
    seg_feats = list(x.columns[:6])
    model_feats = list(x.columns[:3])
    result = compute_segment_oof_probabilities(x, y, seg_feats, model_feats, config)
    assert len(result.oof_probabilities) == len(y)
    assert 0.0 <= result.oof_probabilities.min() <= result.oof_probabilities.max() <= 1.0
    assert result.feature_count == 3


def test_segment_test_probabilities_shape() -> None:
    x, y, x_test = _toy_data()
    config = SegmentConfig(n_clusters=3, min_cluster_samples=20, cv_folds=3)
    seg_feats = list(x.columns[:6])
    model_feats = list(x.columns[:3])
    probs = predict_segment_test_probabilities(x, y, x_test, seg_feats, model_feats, config)
    assert probs.shape == (len(x_test),)
