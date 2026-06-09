"""Shared helpers for modeling notebooks."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .dataset import get_classifier, load_test_data, load_training_data
from .models import make_tuned_factory, run_hyperparameter_search
from .models.modeling import build_top_k_feature_sets

DEFAULT_STAGE3_TOP_K_SIZES: tuple[int, ...] = (1, 3, 4, 5, 8, 10, 15, 20)

DEFAULT_SUBMISSION_PREFIX = "pozorski_florek_poltorak"
MODELING_CV_FOLDS = 5
HPO_N_ITER = 20

LOGISTIC_PARAM_GRID: dict[str, list] = {
    "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "logisticregression__penalty": ["l1", "l2"],
}

LGB_PARAM_DIST: dict[str, list] = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 4, 6, 8, -1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

XGB_PARAM_DIST: dict[str, list] = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

EstimatorFactory = Callable[[np.ndarray], Any]


@dataclass(frozen=True, slots=True)
class ModelingStageData:
    """Train/test matrices and Stage 3 feature subsets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    X_stage2: pd.DataFrame
    stage2_features: tuple[str, ...]
    feature_set_candidates: dict[str, list[str]]
    stage3_scores: pd.DataFrame


def configure_notebook_style(random_state: int = 42) -> None:
    """Apply shared matplotlib/seaborn defaults for notebooks."""
    np.random.seed(random_state)
    sns.set_theme(style="darkgrid", palette="husl")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["font.size"] = 11


def var_name_to_index(name: str) -> int:
    """Convert ``var_123`` to integer index ``123``."""
    match = re.fullmatch(r"var_(\d+)", name)
    if not match:
        raise ValueError(f"Unexpected feature name: {name}")
    return int(match.group(1))


def feature_set_candidates_from_selection_results(
    feature_selection_outputs: Path,
    sizes: tuple[int, ...] | None = None,
) -> dict[str, list[str]]:
    """Build ``top_01``, ``top_03``, … from ``feature_selection_results.csv`` ranking."""
    path = feature_selection_outputs / "feature_selection_results.csv"
    ranked = pd.read_csv(path).sort_values("order")
    features = ranked["feature"].tolist()
    if not features:
        msg = f"No ranked features in {path}"
        raise ValueError(msg)
    if sizes is None:
        sizes = (*DEFAULT_STAGE3_TOP_K_SIZES, len(features))
    return build_top_k_feature_sets(features, sizes=sizes)


def load_modeling_stage_data(
    data_path: Path,
    feature_selection_outputs: Path,
    exclude_top_26: bool = False,
) -> ModelingStageData:
    """Load train/test and Stage 3 feature-set candidates."""
    x_train, y_train = load_training_data(data_path)
    x_test = load_test_data(data_path)

    stage3_scores = pd.read_csv(feature_selection_outputs / "stage3_feature_set_scores.csv")
    stage3_scores["features"] = stage3_scores["features"].apply(ast.literal_eval)

    stage2_row = stage3_scores.loc[stage3_scores["feature_count"].idxmax()]
    stage2_features = tuple(stage2_row["features"])
    feature_set_candidates = {
        row.feature_set_name: row.features
        for row in stage3_scores.itertuples(index=False)
        if not (exclude_top_26 and row.feature_set_name == "top_26")
    }

    return ModelingStageData(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        X_stage2=x_train[list(stage2_features)],
        stage2_features=stage2_features,
        feature_set_candidates=feature_set_candidates,
        stage3_scores=stage3_scores,
    )


def write_submission_files(
    outputs_path: Path,
    submission_prefix: str,
    customer_indices: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
) -> tuple[Path, Path]:
    """Write ``{prefix}_obs.txt`` and ``{prefix}_vars.txt`` submission files."""
    obs_path = outputs_path / f"{submission_prefix}_obs.txt"
    vars_path = outputs_path / f"{submission_prefix}_vars.txt"

    obs_path.write_text(
        "\n".join(str(int(i) + 1) for i in customer_indices) + "\n",
        encoding="utf-8",
    )
    vars_path.write_text(
        "\n".join(str(var_name_to_index(name) + 1) for name in feature_names) + "\n",
        encoding="utf-8",
    )
    return obs_path, vars_path


def persist_hpo_results(
    outputs_path: Path,
    model_name: str,
    best_params: dict[str, Any],
    hpo_cv: float,
) -> Path:
    """Save per-model and summary HPO JSON under ``outputs_path/hpo/``."""
    hpo_dir = outputs_path / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    payload = {"best_params": best_params, "cv_business_mean": hpo_cv}
    with (hpo_dir / f"{model_name}_hpo.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with (hpo_dir / "hpo_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({model_name: payload}, fh, indent=2)
    return hpo_dir


def run_winner_hyperparameter_search(
    model_name: str,
    estimator_factory: EstimatorFactory,
    X_best: pd.DataFrame,
    y_train: pd.Series,
    outputs_path: Path,
    cv_folds: int = MODELING_CV_FOLDS,
    hpo_n_iter: int = HPO_N_ITER,
    logistic_param_grid: dict[str, list] | None = None,
    lgb_param_dist: dict[str, list] | None = None,
    xgb_param_dist: dict[str, list] | None = None,
) -> tuple[Any, dict[str, Any], float, EstimatorFactory]:
    """Tune the winning model and persist HPO artifacts."""
    logistic_param_grid = logistic_param_grid or LOGISTIC_PARAM_GRID
    lgb_param_dist = lgb_param_dist or LGB_PARAM_DIST
    xgb_param_dist = xgb_param_dist or XGB_PARAM_DIST

    if model_name == "logistic_regression":
        best_est, best_params, hpo_cv = run_hyperparameter_search(
            estimator_factory(y_train.to_numpy()),
            X_best,
            y_train,
            param_grid=logistic_param_grid,
            cv=cv_folds,
        )
    elif model_name == "lightgbm":
        best_est, best_params, hpo_cv = run_hyperparameter_search(
            get_classifier(y_train.to_numpy()),
            X_best,
            y_train,
            param_dist=lgb_param_dist,
            n_iter=hpo_n_iter,
            cv=cv_folds,
        )
    else:
        best_est, best_params, hpo_cv = run_hyperparameter_search(
            estimator_factory(y_train.to_numpy()),
            X_best,
            y_train,
            param_dist=xgb_param_dist,
            n_iter=hpo_n_iter,
            cv=cv_folds,
        )

    persist_hpo_results(outputs_path, model_name, best_params, hpo_cv)
    return best_est, best_params, hpo_cv, make_tuned_factory(best_est)
