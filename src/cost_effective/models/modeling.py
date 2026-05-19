"""Modeling helpers for feature-set comparison and threshold tuning."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..dataset.utils import custom_scorer, get_classifier


EstimatorFactory = Callable[[np.ndarray], Any]


@dataclass(frozen=True, slots=True)
class FeatureSetEvaluation:
    """Cross-validated score for a single feature subset."""

    feature_set_name: str
    feature_count: int
    cv_score_mean: float
    cv_score_std: float
    features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "feature_set_name": self.feature_set_name,
            "feature_count": self.feature_count,
            "cv_score_mean": self.cv_score_mean,
            "cv_score_std": self.cv_score_std,
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "model_name": self.model_name,
            "feature_set_name": self.feature_set_name,
            "feature_count": self.feature_count,
            "cv_score_mean": self.cv_score_mean,
            "cv_score_std": self.cv_score_std,
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


def rank_features(feature_names: list[str], rankings: np.ndarray) -> pd.DataFrame:
    """Return features ordered by RFECV rank, then name as a stable tiebreaker."""
    ranked = pd.DataFrame({"feature": feature_names, "ranking": np.asarray(rankings, dtype=int)})
    ranked = ranked.sort_values(["ranking", "feature"], ascending=[True, True]).reset_index(drop=True)
    ranked["order"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_top_k_feature_sets(
    ranked_features: list[str] | pd.Series | pd.Index,
    sizes: list[int] | tuple[int, ...] | None = None,
) -> dict[str, list[str]]:
    """Create named top-k feature subsets from a ranked feature list."""
    feature_list = list(ranked_features)
    if not feature_list:
        return {}

    if sizes is None:
        candidate_sizes = {1, 3, 5, 8, 10, 15, 20, len(feature_list)}
    else:
        candidate_sizes = {size for size in sizes if size > 0}
        candidate_sizes.add(len(feature_list))

    sorted_sizes = sorted(size for size in candidate_sizes if size <= len(feature_list))
    return {f"top_{size:02d}": feature_list[:size] for size in sorted_sizes}


def evaluate_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict[str, list[str]],
    estimator_factory: EstimatorFactory | None = None,
    cv: int = 5,
) -> pd.DataFrame:
    """Score feature subsets with the business metric under cross-validation."""
    factory = estimator_factory or get_classifier
    evaluations: list[FeatureSetEvaluation] = []

    for feature_set_name, features in feature_sets.items():
        if not features:
            continue

        scores = cross_val_score(
            factory(y.to_numpy()),
            X[features],
            y,
            cv=cv,
            scoring=custom_scorer,
        )
        evaluations.append(
            FeatureSetEvaluation(
                feature_set_name=feature_set_name,
                feature_count=len(features),
                cv_score_mean=float(scores.mean()),
                cv_score_std=float(scores.std(ddof=0)),
                features=tuple(features),
            )
        )

    return pd.DataFrame([evaluation.to_dict() for evaluation in evaluations]).sort_values(
        ["cv_score_mean", "feature_count"], ascending=[False, True]
    ).reset_index(drop=True)


def build_model_factories(y: pd.Series) -> dict[str, EstimatorFactory]:
    """Return a small model zoo for comparison on ranked feature sets."""

    y_array = y.to_numpy()

    def _lightgbm_factory(_: np.ndarray | None = None) -> Any:
        return get_classifier(y=y_array)

    def _logistic_factory(_: np.ndarray | None = None) -> Any:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42,
            ),
        )

    factories: dict[str, EstimatorFactory] = {
        "lightgbm": _lightgbm_factory,
        "logistic_regression": _logistic_factory,
    }

    try:
        import xgboost as xgb  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:

        def _xgboost_factory(_: np.ndarray | None = None) -> Any:
            return xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
            )

        factories["xgboost"] = _xgboost_factory

    return factories


def compare_models_on_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict[str, list[str]],
    estimator_factories: dict[str, EstimatorFactory] | None = None,
    cv: int = 5,
) -> pd.DataFrame:
    """Evaluate multiple model families on the same ranked feature subsets."""
    factories = estimator_factories or build_model_factories(y)
    comparisons: list[ModelComparisonResult] = []

    for model_name, factory in factories.items():
        for feature_set_name, features in feature_sets.items():
            if not features:
                continue

            scores = cross_val_score(
                factory(y.to_numpy()),
                X[features],
                y,
                cv=cv,
                scoring=custom_scorer,
            )
            comparisons.append(
                ModelComparisonResult(
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    feature_count=len(features),
                    cv_score_mean=float(scores.mean()),
                    cv_score_std=float(scores.std(ddof=0)),
                )
            )

    return pd.DataFrame([comparison.to_dict() for comparison in comparisons]).sort_values(
        ["cv_score_mean", "feature_count"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_oof_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_factory: EstimatorFactory | None = None,
    cv: int = 5,
) -> np.ndarray:
    """Generate out-of-fold positive-class probabilities."""
    factory = estimator_factory or get_classifier
    estimator = factory(y.to_numpy())
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        estimator,
        X,
        y,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return probabilities


def build_profit_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    feature_count: int,
    max_targets: int = 1000,
) -> ProfitCurveResult:
    """Create a profit curve from ranked probabilities and feature cost."""
    y_array = y_true.to_numpy()
    ranking = np.argsort(probabilities)[::-1]
    ranked_targets = y_array[ranking]
    ranked_probabilities = probabilities[ranking]

    limit = min(max_targets, len(ranked_targets))
    cumulative_tp = np.cumsum(ranked_targets[:limit] == 1)
    cumulative_fp = np.cumsum(ranked_targets[:limit] == 0)
    scores = (cumulative_tp * 10) - (cumulative_fp * 5) - (feature_count * 200)

    curve = pd.DataFrame(
        {
            "k": np.arange(1, limit + 1),
            "tp": cumulative_tp,
            "fp": cumulative_fp,
            "score": scores,
            "threshold": ranked_probabilities[:limit],
        }
    )

    best_idx = int(curve["score"].idxmax())
    best_row = curve.iloc[best_idx]
    return ProfitCurveResult(
        curve=curve,
        best_k=int(best_row["k"]),
        best_threshold=float(best_row["threshold"]),
        best_score=float(best_row["score"]),
    )


def fit_final_model_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    selected_features: list[str],
    estimator_factory: EstimatorFactory | None = None,
    max_targets: int = 1000,
) -> FinalPredictionResult:
    """Fit the final model on the selected feature subset and rank test samples."""
    factory = estimator_factory or get_classifier
    model = factory(y_train.to_numpy())
    model.fit(X_train[selected_features], y_train)
    probabilities = model.predict_proba(X_test[selected_features])[:, 1]
    ranked_test_indices = np.argsort(probabilities)[::-1][: min(max_targets, len(probabilities))]
    threshold = float(probabilities[ranked_test_indices[-1]]) if len(ranked_test_indices) else 0.0

    return FinalPredictionResult(
        model_name=model.__class__.__name__,
        selected_features=tuple(selected_features),
        probabilities=probabilities,
        ranked_test_indices=ranked_test_indices,
        threshold=threshold,
    )
