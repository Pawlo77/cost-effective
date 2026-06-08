"""Tests for baseline notebook helpers."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cost_effective.models.baselines import (
    RandomScoreClassifier,
    best_univariate_feature,
    cross_validate_baseline,
    make_dummy_baseline,
    zero_feature_matrix,
)


def test_zero_feature_matrix_shape() -> None:
    x = zero_feature_matrix(50)
    assert x.shape == (50, 0)


def test_random_score_classifier_proba_shape() -> None:
    x = zero_feature_matrix(20)
    y = np.array([0, 1] * 10)
    clf = RandomScoreClassifier(random_state=0).fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (20, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_cross_validate_baseline_prior_dummy() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    x = zero_feature_matrix(len(y))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    result = cross_validate_baseline(
        "prior",
        make_dummy_baseline("prior"),
        x,
        y,
        cv=cv,
        n_jobs=1,
    )
    assert result["n_features"] == 0
    assert "business_no_var_penalty" in result["summary"]
    assert len(result["summary"]["business_no_var_penalty"]["folds"]) == 3


def test_best_univariate_feature() -> None:
    x = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 1, 1, 1]})
    y = pd.Series([0, 1, 0, 1])
    assert best_univariate_feature(x, y) in {"a", "b"}
