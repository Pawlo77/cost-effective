"""Reusable model-family specifications and factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

EstimatorFactory = Callable[[np.ndarray], Any]


@dataclass(frozen=True, slots=True)
class ModelFamilySpec:
    """Specification for a classifier or ranker family used in OOF validation."""

    model_family: str
    kind: str
    factory: EstimatorFactory | None = None
    params: Mapping[str, Any] | None = None
    max_sets: int | None = None


def scale_pos_weight(y_values: np.ndarray) -> float:
    """Compute negative/positive ratio for imbalanced binary classifiers."""
    y_values = np.asarray(y_values)
    positives = int((y_values == 1).sum())
    negatives = int((y_values == 0).sum())
    return negatives / positives if positives > 0 else 1.0


def weighted_scale_pos_weight(
    y_values: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    """Resolve explicit or multiplier-based positive-class weight."""
    if "scale_pos_weight" in params:
        return float(params["scale_pos_weight"])
    multiplier = float(params.get("scale_pos_weight_multiplier", 1.0))
    return scale_pos_weight(y_values) * multiplier


def positive_scores(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return positive-class scores from a classifier-like model."""
    if hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict_proba(X))
        return pred[:, 1] if pred.ndim == 2 else pred
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def logistic_factory(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> EstimatorFactory:
    """Factory for a robust-scaled elastic-net logistic baseline."""
    params = dict(params or {})
    return lambda _y: make_pipeline(
        RobustScaler(),
        LogisticRegression(
            max_iter=int(params.get("max_iter", 1500)),
            solver=str(params.get("solver", "saga")),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=random_state,
        ),
    )


def lightgbm_factory(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> EstimatorFactory:
    """Factory for a compact LightGBM classifier."""
    params = dict(params or {})

    def _factory(y_values: np.ndarray) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 120)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 24)),
            min_child_samples=int(params.get("min_child_samples", 25)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=weighted_scale_pos_weight(y_values, params),
            random_state=random_state,
            verbose=-1,
            n_jobs=1,
        )

    return _factory


def xgboost_factory(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> EstimatorFactory:
    """Factory for a compact XGBoost classifier."""
    import xgboost as xgb

    params = dict(params or {})

    def _factory(y_values: np.ndarray) -> Any:
        return xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 120)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=weighted_scale_pos_weight(y_values, params),
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        )

    return _factory


def random_forest_factory(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> EstimatorFactory:
    """Factory for a small RandomForest sanity-check model."""
    params = dict(params or {})
    return lambda _y: RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 160)),
        max_depth=params.get("max_depth", 5),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        class_weight=params.get("class_weight", "balanced_subsample"),
        random_state=random_state,
        n_jobs=1,
    )


def extra_trees_factory(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> EstimatorFactory:
    """Factory for a small ExtraTrees sanity-check model."""
    params = dict(params or {})
    return lambda _y: ExtraTreesClassifier(
        n_estimators=int(params.get("n_estimators", 180)),
        max_depth=params.get("max_depth", 5),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        class_weight=params.get("class_weight", "balanced"),
        random_state=random_state,
        n_jobs=1,
    )


def ebm_factory(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> EstimatorFactory:
    """Factory for an additive Explainable Boosting Machine."""
    from interpret.glassbox import ExplainableBoostingClassifier

    params = dict(params or {})
    return lambda _y: ExplainableBoostingClassifier(
        interactions=params.get("interactions", 0),
        max_bins=int(params.get("max_bins", 64)),
        outer_bags=int(params.get("outer_bags", 4)),
        inner_bags=int(params.get("inner_bags", 0)),
        max_rounds=int(params.get("max_rounds", 3000)),
        early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
        random_state=random_state,
        n_jobs=1,
    )


def make_classifier_spec(
    model_family: str,
    params: Mapping[str, Any] | None = None,
    *,
    random_state: int = 42,
    max_sets: int | None = None,
) -> ModelFamilySpec:
    """Build a classifier model-family spec by name."""
    factories = {
        "logistic_baseline": logistic_factory,
        "lightgbm_classifier": lightgbm_factory,
        "xgboost_classifier": xgboost_factory,
        "random_forest_small": random_forest_factory,
        "extra_trees_small": extra_trees_factory,
        "ebm_additive": ebm_factory,
    }
    if model_family not in factories:
        raise ValueError(f"Unknown classifier model family: {model_family}")
    return ModelFamilySpec(
        model_family=model_family,
        kind="classifier",
        factory=factories[model_family](params, random_state=random_state),
        params=dict(params or {}),
        max_sets=max_sets,
    )


def make_lambdamart_spec(
    params: Mapping[str, Any] | None = None,
    *,
    max_sets: int | None = None,
) -> ModelFamilySpec:
    """Build a LambdaMART ranker model spec."""
    return ModelFamilySpec(
        model_family="lambdamart_ranker",
        kind="lambdamart",
        params=dict(params or {}),
        max_sets=max_sets,
    )


def default_model_specs(
    *,
    random_state: int = 42,
    max_sets: int | None = None,
    ebm_max_sets: int | None = None,
    include_xgboost: bool = True,
    include_ebm: bool = True,
    include_lambdamart: bool = True,
) -> list[ModelFamilySpec]:
    """Default model-family validation set."""
    specs = [
        make_classifier_spec("logistic_baseline", random_state=random_state, max_sets=max_sets),
        make_classifier_spec("lightgbm_classifier", random_state=random_state, max_sets=max_sets),
        make_classifier_spec("extra_trees_small", random_state=random_state, max_sets=max_sets),
        make_classifier_spec("random_forest_small", random_state=random_state, max_sets=max_sets),
    ]
    if include_xgboost:
        specs.append(
            make_classifier_spec("xgboost_classifier", random_state=random_state, max_sets=max_sets)
        )
    if include_lambdamart:
        specs.append(make_lambdamart_spec(max_sets=max_sets))
    if include_ebm:
        specs.append(
            make_classifier_spec("ebm_additive", random_state=random_state, max_sets=ebm_max_sets)
        )
    return specs


def classifier_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    estimator_factory: EstimatorFactory,
    *,
    cv: int,
    random_state: int = 42,
) -> np.ndarray:
    """OOF scores for a classifier family."""
    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    feature_list = list(features)
    for train_idx, val_idx in splitter.split(X[feature_list], y):
        model = estimator_factory(y.iloc[train_idx].to_numpy())
        model.fit(X[feature_list].iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = positive_scores(model, X[feature_list].iloc[val_idx])
    return oof


def fit_classifier_full(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    estimator_factory: EstimatorFactory,
) -> Any:
    """Fit a classifier family on all training rows."""
    feature_list = list(features)
    model = estimator_factory(y.to_numpy())
    model.fit(X[feature_list], y)
    return model


def _make_lambdamart(
    params: Mapping[str, Any] | None = None, *, random_state: int = 42
) -> LGBMRanker:
    params = dict(params or {})
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(params.get("n_estimators", 120)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        num_leaves=int(params.get("num_leaves", 24)),
        min_child_samples=int(params.get("min_child_samples", 25)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=random_state,
        verbose=-1,
        n_jobs=1,
    )


def lambdamart_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    params: Mapping[str, Any] | None = None,
    cv: int,
    random_state: int = 42,
) -> np.ndarray:
    """OOF scores for a LambdaMART ranker treating each fold as one group."""
    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    feature_list = list(features)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X[feature_list], y)):
        model = _make_lambdamart(params, random_state=random_state + fold)
        model.fit(
            X[feature_list].iloc[train_idx], y.iloc[train_idx].astype(int), group=[len(train_idx)]
        )
        oof[val_idx] = np.asarray(model.predict(X[feature_list].iloc[val_idx]), dtype=float)
    return oof


def fit_lambdamart_full(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    params: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> LGBMRanker:
    """Fit LambdaMART on all training rows."""
    feature_list = list(features)
    model = _make_lambdamart(params, random_state=random_state)
    model.fit(X[feature_list], y.astype(int), group=[len(y)])
    return model


def small_hpo_grid() -> dict[str, list[dict[str, Any]]]:
    """Small business-score-oriented hyperparameter grids for top candidates."""
    return {
        "extra_trees_small": [
            {
                "n_estimators": 240,
                "max_depth": 4,
                "min_samples_leaf": 12,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 240,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 300,
                "max_depth": 7,
                "min_samples_leaf": 25,
                "class_weight": None,
            },
        ],
        "random_forest_small": [
            {
                "n_estimators": 220,
                "max_depth": 4,
                "min_samples_leaf": 12,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 220,
                "max_depth": 6,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 300,
                "max_depth": 7,
                "min_samples_leaf": 25,
                "class_weight": None,
            },
        ],
        "lightgbm_classifier": [
            {
                "n_estimators": 120,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_child_samples": 15,
                "scale_pos_weight_multiplier": 0.75,
            },
            {
                "n_estimators": 160,
                "learning_rate": 0.04,
                "num_leaves": 24,
                "min_child_samples": 25,
                "scale_pos_weight_multiplier": 1.0,
            },
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 35,
                "scale_pos_weight_multiplier": 1.25,
            },
        ],
        "xgboost_classifier": [
            {
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "scale_pos_weight_multiplier": 0.75,
            },
            {
                "n_estimators": 160,
                "learning_rate": 0.04,
                "max_depth": 3,
                "min_child_weight": 3.0,
                "scale_pos_weight_multiplier": 1.0,
            },
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "scale_pos_weight_multiplier": 1.25,
            },
        ],
        "logistic_baseline": [
            {"C": 0.2, "l1_ratio": 0.2, "class_weight": "balanced"},
            {"C": 0.5, "l1_ratio": 0.5, "class_weight": "balanced"},
            {"C": 1.0, "l1_ratio": 0.8, "class_weight": None},
        ],
    }


def specs_from_hpo_grid(
    grid: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    random_state: int = 42,
) -> list[ModelFamilySpec]:
    """Expand an HPO grid into validation specs."""
    specs: list[ModelFamilySpec] = []
    for model_family, param_list in grid.items():
        for idx, params in enumerate(param_list, start=1):
            spec = make_classifier_spec(model_family, params, random_state=random_state)
            specs.append(
                ModelFamilySpec(
                    model_family=f"{model_family}_hpo{idx:02d}",
                    kind=spec.kind,
                    factory=spec.factory,
                    params=dict(params),
                    max_sets=spec.max_sets,
                )
            )
    return specs
