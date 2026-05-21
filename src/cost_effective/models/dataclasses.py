"""Data containers for modeling results."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureSetEvaluation:
    """Cross-validated score for a single feature subset."""

    feature_set_name: str
    feature_count: int
    cv_score_mean: float
    cv_score_std: float
    features: tuple[str, ...]
    f1_score: float | None = None
    business_score_no_var_penalty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "feature_set_name": self.feature_set_name,
            "feature_count": self.feature_count,
            "cv_score_mean": self.cv_score_mean,
            "cv_score_std": self.cv_score_std,
            "f1_score": self.f1_score,
            "business_score_no_var_penalty": self.business_score_no_var_penalty,
            "features": list(self.features),
        }


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """Cross-validated performance for a model/feature-set pair."""

    model_name: str
    feature_set_name: str
    feature_count: int
    cv_score_mean: float
    cv_score_std: float
    f1_score: float | None = None
    business_score_no_var_penalty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "model_name": self.model_name,
            "feature_set_name": self.feature_set_name,
            "feature_count": self.feature_count,
            "cv_score_mean": self.cv_score_mean,
            "cv_score_std": self.cv_score_std,
            "f1_score": self.f1_score,
            "business_score_no_var_penalty": self.business_score_no_var_penalty,
        }


@dataclass(frozen=True, slots=True)
class ProfitCurveResult:
    """Business-score curve built from out-of-fold probabilities."""

    curve: pd.DataFrame
    best_k: int
    best_threshold: float
    best_score: float


@dataclass(frozen=True, slots=True)
class FinalPredictionResult:
    """Final model and ranked test predictions."""

    model_name: str
    selected_features: tuple[str, ...]
    probabilities: np.ndarray
    ranked_test_indices: np.ndarray
    threshold: float
