"""Rank-fusion committee: diverse experts, weighted OOF probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_is_fitted

from ..dataset.utils import custom_scorer
from .modeling import build_model_factories, compute_oof_probabilities


@dataclass(frozen=True, slots=True)
class FusionExpertConfig:
    """One committee member: feature subset + model family."""

    name: str
    feature_set_name: str
    model_name: str


DEFAULT_FUSION_EXPERTS: tuple[FusionExpertConfig, ...] = (
    FusionExpertConfig("logistic_top03", "top_03", "logistic_regression"),
    FusionExpertConfig("lightgbm_top05", "top_05", "lightgbm"),
    FusionExpertConfig("logistic_top08", "top_08", "logistic_regression"),
    FusionExpertConfig("borda_top05", "top_05", "borda_rank"),
)


class BordaRankClassifier(BaseEstimator, ClassifierMixin):
    """Average rank across features (direction from train correlation sign)."""

    def __init__(self, feature_names: tuple[str, ...] | list[str] | None = None) -> None:
        if feature_names is None:
            self.feature_names: tuple[str, ...] = ()
        elif isinstance(feature_names, tuple):
            self.feature_names = feature_names
        else:
            self.feature_names = tuple(feature_names)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> BordaRankClassifier:
        columns = list(self.feature_names)
        if isinstance(X, pd.DataFrame):
            matrix = X[columns].to_numpy(dtype=float)
        else:
            matrix = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_features_in_ = len(self.feature_names)
        self.classes_ = np.array([0, 1])
        signs = np.zeros(matrix.shape[1])
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            if np.std(col) < 1e-12:
                signs[j] = 1.0
            else:
                signs[j] = float(np.sign(np.corrcoef(col, y)[0, 1]) or 1.0)
        self.signs_ = signs
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "signs_")
        columns = list(self.feature_names)
        if isinstance(X, pd.DataFrame):
            matrix = X[columns].to_numpy(dtype=float)
        else:
            matrix = np.asarray(X, dtype=float)
        n = matrix.shape[0]
        score = np.zeros(n, dtype=float)
        for j in range(matrix.shape[1]):
            directed = self.signs_[j] * matrix[:, j]
            order = np.argsort(directed)
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(1, n + 1, dtype=float)
            score += (n + 1) - ranks
        score /= max(matrix.shape[1], 1)
        prob = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return np.column_stack([1.0 - prob, prob])


@dataclass(frozen=True, slots=True)
class FusionExpertResult:
    """OOF probabilities and CV business score for one expert."""

    config: FusionExpertConfig
    features: tuple[str, ...]
    oof_probabilities: np.ndarray
    cv_business_mean: float
    cv_business_std: float
    weight: float


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Fused OOF scores and expert metadata."""

    experts: tuple[FusionExpertResult, ...]
    fused_oof_probabilities: np.ndarray
    submission_features: tuple[str, ...]
    feature_count: int
    weights: dict[str, float]


def _expert_factory(
    model_name: str,
    features: list[str],
    y: np.ndarray,
    factories: dict[str, Any],
) -> Any:
    if model_name == "borda_rank":
        return BordaRankClassifier(tuple(features))
    return factories[model_name](y)


def evaluate_fusion_expert(
    X: pd.DataFrame,
    y: pd.Series,
    config: FusionExpertConfig,
    features: list[str],
    factories: dict[str, Any],
    cv: int = 5,
) -> FusionExpertResult:
    """CV business score and OOF probabilities for one expert."""
    y_array = y.to_numpy()
    estimator = _expert_factory(config.model_name, features, y_array, factories)
    cv_results = cross_validate(
        estimator,
        X[features],
        y,
        cv=cv,
        scoring={"business": custom_scorer},
        n_jobs=-1,
    )
    oof = compute_oof_probabilities(
        X[features],
        y,
        estimator_factory=lambda _: _expert_factory(
            config.model_name, features, y_array, factories
        ),
        cv=cv,
    )
    scores = cv_results["test_business"]
    return FusionExpertResult(
        config=config,
        features=tuple(features),
        oof_probabilities=oof,
        cv_business_mean=float(scores.mean()),
        cv_business_std=float(scores.std(ddof=0)),
        weight=0.0,
    )


def fusion_weights_from_cv(
    experts: list[FusionExpertResult],
    *,
    min_weight: float = 0.0,
) -> dict[str, float]:
    """Non-negative weights proportional to CV business (zero floor)."""
    raw = {e.config.name: max(min_weight, e.cv_business_mean) for e in experts}
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(experts) if experts else 0.0
        return {e.config.name: uniform for e in experts}
    return {name: value / total for name, value in raw.items()}


def fuse_probabilities(
    experts: list[FusionExpertResult],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted average of expert OOF or test probabilities."""
    if not experts:
        raise ValueError("At least one expert is required")
    fused = np.zeros_like(experts[0].oof_probabilities, dtype=float)
    for expert in experts:
        w = weights[expert.config.name]
        fused += w * expert.oof_probabilities
    return fused


def union_expert_features(experts: list[FusionExpertResult]) -> tuple[str, ...]:
    """Sorted union of all expert feature columns for submission."""
    names: set[str] = set()
    for expert in experts:
        names.update(expert.features)
    return tuple(sorted(names))


def run_fusion_committee(
    X: pd.DataFrame,
    y: pd.Series,
    feature_set_candidates: dict[str, list[str]],
    expert_configs: tuple[FusionExpertConfig, ...] | None = None,
    cv: int = 5,
) -> FusionResult:
    """Evaluate all experts, fuse OOF probabilities, return submission feature union."""
    specs = expert_configs if expert_configs is not None else DEFAULT_FUSION_EXPERTS
    factories = build_model_factories(y)
    results: list[FusionExpertResult] = []
    for spec in specs:
        if spec.feature_set_name not in feature_set_candidates:
            continue
        features = feature_set_candidates[spec.feature_set_name]
        if not features:
            continue
        expert = evaluate_fusion_expert(X, y, spec, features, factories, cv=cv)
        results.append(expert)

    if not results:
        raise ValueError("No fusion experts could be built from feature_set_candidates")

    weights = fusion_weights_from_cv(results)
    experts_with_weights = [
        FusionExpertResult(
            config=e.config,
            features=e.features,
            oof_probabilities=e.oof_probabilities,
            cv_business_mean=e.cv_business_mean,
            cv_business_std=e.cv_business_std,
            weight=weights[e.config.name],
        )
        for e in results
    ]
    fused = fuse_probabilities(experts_with_weights, weights)
    submission_features = union_expert_features(experts_with_weights)
    return FusionResult(
        experts=tuple(experts_with_weights),
        fused_oof_probabilities=fused,
        submission_features=submission_features,
        feature_count=len(submission_features),
        weights=weights,
    )


def predict_fusion_test_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    fusion: FusionResult,
    factories: dict[str, Any] | None = None,
) -> np.ndarray:
    """Fit each expert on full train and return fused test positive probabilities."""
    y_array = y_train.to_numpy()
    model_factories = factories or build_model_factories(y_train)
    test_fused = np.zeros(len(X_test), dtype=float)
    for expert in fusion.experts:
        features = list(expert.features)
        estimator = _expert_factory(
            expert.config.model_name,
            features,
            y_array,
            model_factories,
        )
        estimator.fit(X_train[features], y_train)
        test_prob = estimator.predict_proba(X_test[features])[:, 1]
        test_fused += fusion.weights[expert.config.name] * test_prob
    return test_fused


def experts_summary_frame(fusion: FusionResult) -> pd.DataFrame:
    """Table of per-expert CV metrics and fusion weights."""
    rows = [
        {
            "expert": e.config.name,
            "model": e.config.model_name,
            "feature_set": e.config.feature_set_name,
            "n_features": len(e.features),
            "cv_business_mean": e.cv_business_mean,
            "cv_business_std": e.cv_business_std,
            "fusion_weight": e.weight,
        }
        for e in fusion.experts
    ]
    return pd.DataFrame(rows).sort_values("fusion_weight", ascending=False)
