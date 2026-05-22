"""Modeling helpers for feature-set comparison and threshold tuning."""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from ..dataset.utils import (
    DEFAULT_MAX_TARGETS,
    best_k_break_even,
    business_scorer_no_var_penalty,
    custom_scorer,
    f1_scorer_wrapper,
    get_classifier,
)
from .dataclasses import (
    F1CurveResult,
    FeatureSetEvaluation,
    FinalPredictionResult,
    ModelComparisonResult,
    ProfitCurveResult,
)

EstimatorFactory = Callable[[np.ndarray], Any]


def rank_features_drop_column_cv(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_factory: EstimatorFactory | None = None,
    cv: int = 5,
) -> pd.DataFrame:
    """Rank features by CV business impact when each column is removed.

    Lower ``delta`` (score drop when removed) means higher importance.
    """
    factory = estimator_factory or get_classifier
    y_array = y.to_numpy()
    cv_scoring = {"business": custom_scorer}
    baseline = float(
        cross_validate(
            factory(y_array),
            X,
            y,
            cv=cv,
            scoring=cv_scoring,
            n_jobs=-1,
        )["test_business"].mean()
    )

    rows: list[dict[str, float | str]] = []
    for feature in X.columns:
        remaining = [col for col in X.columns if col != feature]
        score_without = float(
            cross_validate(
                factory(y_array),
                X[remaining],
                y,
                cv=cv,
                scoring=cv_scoring,
                n_jobs=-1,
            )["test_business"].mean()
        )
        rows.append({
            "feature": feature,
            "cv_score_if_dropped": score_without,
            "delta": score_without - baseline,
        })

    ranked = pd.DataFrame(rows).sort_values(["delta", "feature"], ascending=[True, True])
    ranked = ranked.reset_index(drop=True)
    ranked["order"] = np.arange(1, len(ranked) + 1)
    return ranked


def rank_features(feature_names: list[str], rankings: np.ndarray) -> pd.DataFrame:
    """Return features ordered by rank array (1 = best), then name as tiebreaker."""
    ranked = pd.DataFrame({"feature": feature_names, "ranking": np.asarray(rankings, dtype=int)})
    ranked = ranked.sort_values(["ranking", "feature"], ascending=[True, True])
    ranked = ranked.reset_index(drop=True)
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

        cv_results = cross_validate(
            factory(y.to_numpy()),
            X[features],
            y,
            cv=cv,
            scoring={
                "business": custom_scorer,
                "f1": f1_scorer_wrapper,
                "roc_auc": "roc_auc",
                "business_no_var": business_scorer_no_var_penalty,
            },
            n_jobs=-1,
        )

        cv_scores = cv_results["test_business"]
        f1_scores = cv_results["test_f1"]
        # roc_auc may be absent or NaN if a CV fold contains only one class.
        # Fall back to an OOF-based ROC AUC when necessary.
        roc_auc_scores = cv_results.get("test_roc_auc", np.full(len(cv_scores), np.nan))
        if np.all(np.isnan(roc_auc_scores)):
            try:
                oof = compute_oof_probabilities(X[features], y, estimator_factory=factory, cv=cv)
                roc_val = float(roc_auc_score(y.to_numpy(), oof))
                roc_auc_scores = np.full(len(cv_scores), roc_val)
            except Exception:  # noqa: S110
                # if fallback fails, keep NaNs
                pass
        business_no_var_scores = cv_results.get(
            "test_business_no_var", np.full(len(cv_scores), np.nan)
        )

        evaluations.append(
            FeatureSetEvaluation(
                feature_set_name=feature_set_name,
                feature_count=len(features),
                cv_score_mean=float(cv_scores.mean()),
                cv_score_std=float(cv_scores.std(ddof=0)),
                f1_score=float(f1_scores.mean()),
                roc_auc_score=float(roc_auc_scores.mean()),
                business_score_no_var_penalty=float(business_no_var_scores.mean()),
                features=tuple(features),
            )
        )

    return (
        pd.DataFrame([evaluation.to_dict() for evaluation in evaluations])
        .sort_values(["cv_score_mean", "feature_count"], ascending=[False, True])
        .reset_index(drop=True)
    )


def make_logistic_baseline_pipeline(C: float = 1.0) -> Any:
    """Logistic baseline: per-fold ``RobustScaler`` + balanced LR (for notebooks)."""
    return make_pipeline(
        RobustScaler(),
        LogisticRegression(
            max_iter=2000,
            C=C,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
    )


def build_model_factories(y: pd.Series) -> dict[str, EstimatorFactory]:
    """Return a small model zoo for comparison on ranked feature sets."""

    y_array = y.to_numpy()

    def _lightgbm_factory(_: np.ndarray | None = None) -> Any:
        return get_classifier(y=y_array)

    def _logistic_factory(_: np.ndarray | None = None) -> Any:
        return make_pipeline(
            RobustScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.5,
                class_weight="balanced",
                random_state=42,
            ),
        )

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

    return {
        "lightgbm": _lightgbm_factory,
        "logistic_regression": _logistic_factory,
        "xgboost": _xgboost_factory,
    }


def compare_models_on_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict[str, list[str]],
    estimator_factories: dict[str, EstimatorFactory] | None = None,
    cv: int = 5,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> pd.DataFrame:
    """Evaluate multiple model families on the same ranked feature subsets.

    All business metrics use top-k ranking capped at ``max_targets``.
    """
    factories = estimator_factories or build_model_factories(y)
    comparisons: list[ModelComparisonResult] = []

    def _business_scorer(estimator, X_val: np.ndarray, y_true: np.ndarray) -> float:
        return custom_scorer(estimator, X_val, y_true, max_k=max_targets)

    def _business_no_var_scorer(estimator, X_val: np.ndarray, y_true: np.ndarray) -> float:
        return business_scorer_no_var_penalty(estimator, X_val, y_true, max_k=max_targets)

    def _f1_scorer(estimator, X_val: np.ndarray, y_true: np.ndarray) -> float:
        return f1_scorer_wrapper(estimator, X_val, y_true, max_k=max_targets)

    scorers = {
        "business": _business_scorer,
        "f1": _f1_scorer,
        "roc_auc": "roc_auc",
        "business_no_var": _business_no_var_scorer,
    }

    for model_name, factory in factories.items():
        for feature_set_name, features in feature_sets.items():
            if not features:
                continue

            cv_results = cross_validate(
                factory(y.to_numpy()),
                X[features],
                y,
                cv=cv,
                scoring=scorers,
                n_jobs=-1,
            )

            cv_scores = cv_results["test_business"]
            f1_scores = cv_results["test_f1"]
            roc_auc_scores = cv_results.get("test_roc_auc", np.full(len(cv_scores), np.nan))
            if np.all(np.isnan(roc_auc_scores)):
                try:
                    oof = compute_oof_probabilities(
                        X[features], y, estimator_factory=factory, cv=cv
                    )
                    roc_val = float(roc_auc_score(y.to_numpy(), oof))
                    roc_auc_scores = np.full(len(cv_scores), roc_val)
                except Exception:  # noqa: S110
                    pass
            business_no_var_scores = cv_results.get(
                "test_business_no_var", np.full(len(cv_scores), np.nan)
            )

            comparisons.append(
                ModelComparisonResult(
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    feature_count=len(features),
                    cv_score_mean=float(cv_scores.mean()),
                    cv_score_std=float(cv_scores.std(ddof=0)),
                    f1_score=float(f1_scores.mean()),
                    roc_auc_score=float(roc_auc_scores.mean()),
                    business_score_no_var_penalty=float(business_no_var_scores.mean()),
                )
            )

    return (
        pd.DataFrame([comparison.to_dict() for comparison in comparisons])
        .sort_values(["cv_score_mean", "feature_count"], ascending=[False, True])
        .reset_index(drop=True)
    )


def attach_best_hyperparams(
    comparison: pd.DataFrame,
    model_name: str,
    best_params: dict[str, Any],
) -> pd.DataFrame:
    """Add a CSV-safe ``best_hyperparams`` column from HPO results.

    Assigning a dict directly (``df["col"] = params``) makes pandas align on
    index keys and yields NaN for every row.
    """
    out = comparison.copy()
    serialized = json.dumps(best_params, sort_keys=True)
    out["best_hyperparams"] = np.where(
        out["model_name"] == model_name,
        serialized,
        "",
    )
    return out


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
    return cross_val_predict(
        estimator,
        X,
        y,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


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

    curve = pd.DataFrame({
        "k": np.arange(1, limit + 1),
        "tp": cumulative_tp,
        "fp": cumulative_fp,
        "score": scores,
        "threshold": ranked_probabilities[:limit],
    })

    best_idx = int(curve["score"].idxmax())
    best_row = curve.iloc[best_idx]
    profit_k = int(best_row["k"])
    breakeven_k = best_k_break_even(probabilities, max_k=max_targets)
    effective_k = min(profit_k, breakeven_k) if breakeven_k > 0 else profit_k
    effective_row = curve.loc[curve["k"] == effective_k].iloc[0]
    return ProfitCurveResult(
        curve=curve,
        best_k=effective_k,
        best_threshold=float(effective_row["threshold"]),
        best_score=float(effective_row["score"]),
    )


def build_f1_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> F1CurveResult:
    """Create a threshold sweep optimized for F1, with ROC AUC diagnostics."""
    y_array = y_true.to_numpy()
    ranking = np.argsort(probabilities)[::-1]
    ranked_targets = y_array[ranking]
    ranked_probabilities = probabilities[ranking]

    total_positives = int((y_array == 1).sum())
    total_negatives = int((y_array == 0).sum())
    limit = min(max_targets, len(ranked_targets))

    cumulative_tp = np.cumsum(ranked_targets[:limit] == 1)
    cumulative_fp = np.cumsum(ranked_targets[:limit] == 0)
    cumulative_fn = total_positives - cumulative_tp
    cumulative_tn = total_negatives - cumulative_fp

    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp, dtype=float),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    recall = np.divide(
        cumulative_tp,
        total_positives,
        out=np.zeros_like(cumulative_tp, dtype=float),
        where=total_positives > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    accuracy = np.divide(
        cumulative_tp + cumulative_tn,
        len(y_array),
        out=np.zeros_like(cumulative_tp, dtype=float),
        where=len(y_array) > 0,
    )
    specificity = np.divide(
        cumulative_tn,
        total_negatives,
        out=np.zeros_like(cumulative_tn, dtype=float),
        where=total_negatives > 0,
    )

    curve = pd.DataFrame({
        "k": np.arange(1, limit + 1),
        "tp": cumulative_tp,
        "fp": cumulative_fp,
        "fn": cumulative_fn,
        "tn": cumulative_tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "threshold": ranked_probabilities[:limit],
    })

    best_idx = int(curve["f1"].idxmax())
    best_row = curve.iloc[best_idx]

    return F1CurveResult(
        curve=curve,
        best_k=int(best_row["k"]),
        best_threshold=float(best_row["threshold"]),
        best_f1=float(best_row["f1"]),
        roc_auc=float(roc_auc_score(y_array, probabilities)),
        average_precision=float(average_precision_score(y_array, probabilities)),
    )


def fit_final_model_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    selected_features: list[str],
    estimator_factory: EstimatorFactory | None = None,
    max_targets: int = DEFAULT_MAX_TARGETS,
    n_targets: int | None = None,
) -> FinalPredictionResult:
    """Fit the final model and return top test indices by probability.

    ``n_targets`` overrides ``max_targets`` when set (e.g. OOF-optimal k).
    """
    factory = estimator_factory or get_classifier
    model = factory(y_train.to_numpy())
    model.fit(X_train[selected_features], y_train)
    probabilities = model.predict_proba(X_test[selected_features])[:, 1]
    k = n_targets if n_targets is not None else max_targets
    k = min(k, len(probabilities))
    ranked_test_indices = np.argsort(probabilities)[::-1][:k]
    threshold = float(probabilities[ranked_test_indices[-1]]) if len(ranked_test_indices) else 0.0

    return FinalPredictionResult(
        model_name=model.__class__.__name__,
        selected_features=tuple(selected_features),
        probabilities=probabilities,
        ranked_test_indices=ranked_test_indices,
        threshold=threshold,
    )


def run_hyperparameter_search(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict[str, list] | None = None,
    param_dist: dict[str, list] | None = None,
    n_iter: int = 20,
    cv: int = 5,
) -> tuple[Any, dict[str, Any], float]:
    """Tune hyperparameters on the exact feature matrix used at inference."""
    if (param_grid is None) == (param_dist is None):
        raise ValueError("Provide exactly one of param_grid or param_dist")

    search_kw = {
        "scoring": {
            "business": custom_scorer,
            "f1": f1_scorer_wrapper,
            "roc_auc": "roc_auc",
        },
        "refit": "business",
        "cv": cv,
        "n_jobs": -1,
        "verbose": 0,
    }

    if param_grid is not None:
        search = GridSearchCV(estimator, param_grid, **search_kw)
    else:
        search = RandomizedSearchCV(
            estimator,
            param_dist,
            n_iter=n_iter,
            random_state=42,
            **search_kw,
        )

    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def make_tuned_factory(best_estimator: Any) -> EstimatorFactory:
    """Return a factory that clones a fitted estimator template for CV folds."""
    return lambda _y: clone(best_estimator)
