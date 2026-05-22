"""Reference estimators and helpers for the baseline notebook."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.validation import check_is_fitted

from ..dataset.utils import (
    business_scorer_no_var_penalty,
    custom_scorer,
    f1_scorer_wrapper,
)


class RandomScoreClassifier(BaseEstimator, ClassifierMixin):
    """Uniform random P(y=1) per sample — no-skill ranking baseline."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomScoreClassifier:  # noqa: ARG002
        self.n_features_in_ = int(np.asarray(X).shape[1])
        self.classes_ = np.array([0, 1])
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "rng_")
        n = len(X)
        positive = self.rng_.uniform(0.0, 1.0, size=n)
        return np.column_stack([1.0 - positive, positive])


def zero_feature_matrix(n_samples: int) -> np.ndarray:
    """Feature-less design matrix (n_features=0 for penalized scorers)."""
    return np.zeros((n_samples, 0))


def best_univariate_feature(X: pd.DataFrame, y: pd.Series) -> str:
    """Feature with largest absolute correlation to the target (simple pick)."""
    correlations = X.corrwith(y).abs()
    return str(correlations.idxmax())


def make_dummy_baseline(strategy: str, random_state: int = 42) -> DummyClassifier:
    """Sklearn dummy with flat or random probabilities (use with zero-feature X)."""
    return DummyClassifier(strategy=strategy, random_state=random_state)


def default_baseline_scoring() -> dict[str, Any]:
    """CV scorers aligned with modeling notebooks."""
    return {
        "business_with_var_penalty": custom_scorer,
        "business_no_var_penalty": business_scorer_no_var_penalty,
        "f1_topk": f1_scorer_wrapper,
    }


def cross_validate_baseline(
    name: str,
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: StratifiedKFold | int = 5,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run stratified CV and return fold scores plus summary means/stds."""
    cv_results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=default_baseline_scoring(),
        n_jobs=n_jobs,
    )
    summary: dict[str, dict[str, float]] = {}
    for metric in ("business_with_var_penalty", "business_no_var_penalty", "f1_topk"):
        key = f"test_{metric}"
        scores = cv_results[key]
        summary[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": scores.tolist(),
        }
    return {"name": name, "n_features": int(X.shape[1]), "summary": summary}
