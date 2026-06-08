"""Leakage-safe nested feature-selection experiment scaffolding."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cost_effective.dataset.ensemble_feature_selection import prescreen_combo_name
from cost_effective.models.rank_fusion_specs import expanded_hpo_model_specs

DEFAULT_NESTED_CV_PRESCREEN_METHODS: tuple[str, ...] = (
    "mutual_info",
    "abs_corr",
    "logistic_l1",
    "abess_logistic",
    "lightgbm_gain",
    "ebm",
    "sparse_gam_spam",
    "lambdamart",
)

DEFAULT_NESTED_CV_FEATURE_SIZES: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    15,
    20,
    30,
    40,
    50,
)

EXPERIMENT_LOG_COLUMNS: tuple[str, ...] = (
    "event_id",
    "stage",
    "outer_fold",
    "inner_fold",
    "recipe_id",
    "prescreen_recipe_id",
    "feature_size",
    "model_spec_id",
    "model_family",
    "status",
    "error",
    "selected_features",
    "inner_score",
    "outer_score",
    "started_at",
    "finished_at",
    "duration_seconds",
    "notes",
)

# Backward-compatible aliases for older notebooks/scripts.
DEFAULT_HONEST_PRESCREEN_METHODS = DEFAULT_NESTED_CV_PRESCREEN_METHODS
DEFAULT_HONEST_FEATURE_SIZES = DEFAULT_NESTED_CV_FEATURE_SIZES


def make_outer_fold_assignments(
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign each sample to exactly one outer validation fold."""
    y_array = np.asarray(y, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    assignments = np.full(len(y_array), -1, dtype=int)
    for fold_id, (_train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(y_array)), y_array),
        start=1,
    ):
        assignments[val_idx] = fold_id
    if np.any(assignments < 0):
        raise RuntimeError("Some samples were not assigned to an outer fold")
    return pd.DataFrame({"sample_index": np.arange(len(y_array)), "outer_fold": assignments})


def make_inner_fold_assignments(
    y: pd.Series | np.ndarray,
    outer_assignments: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create inner validation folds inside each outer-training partition only."""
    y_array = np.asarray(y, dtype=int)
    rows: list[dict[str, int]] = []
    outer_folds = sorted(outer_assignments["outer_fold"].unique())
    for outer_fold in outer_folds:
        outer_train = outer_assignments.loc[
            ~outer_assignments["outer_fold"].eq(outer_fold),
            "sample_index",
        ].to_numpy(dtype=int)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + int(outer_fold) * 1009,
        )
        inner_labels = np.full(len(outer_train), -1, dtype=int)
        for inner_fold, (_inner_train_pos, inner_val_pos) in enumerate(
            splitter.split(np.zeros(len(outer_train)), y_array[outer_train]),
            start=1,
        ):
            inner_labels[inner_val_pos] = inner_fold
        if np.any(inner_labels < 0):
            raise RuntimeError(f"Some samples lack inner fold assignment for outer {outer_fold}")
        rows.extend(
            {
                "outer_fold": int(outer_fold),
                "sample_index": int(sample_index),
                "inner_fold": int(inner_fold),
            }
            for sample_index, inner_fold in zip(outer_train, inner_labels, strict=True)
        )
    return pd.DataFrame(rows)


def summarize_outer_folds(
    y: pd.Series | np.ndarray,
    outer_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize class balance for outer train/validation partitions."""
    y_array = np.asarray(y, dtype=int)
    rows: list[dict[str, Any]] = []
    for outer_fold in sorted(outer_assignments["outer_fold"].unique()):
        val_idx = outer_assignments.loc[
            outer_assignments["outer_fold"].eq(outer_fold),
            "sample_index",
        ].to_numpy(dtype=int)
        train_idx = outer_assignments.loc[
            ~outer_assignments["outer_fold"].eq(outer_fold),
            "sample_index",
        ].to_numpy(dtype=int)
        rows.append({
            "outer_fold": int(outer_fold),
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "train_positive_count": int(y_array[train_idx].sum()),
            "val_positive_count": int(y_array[val_idx].sum()),
            "train_positive_rate": float(y_array[train_idx].mean()),
            "val_positive_rate": float(y_array[val_idx].mean()),
        })
    return pd.DataFrame(rows)


def summarize_inner_folds(
    y: pd.Series | np.ndarray,
    inner_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize class balance for nested inner train/validation partitions."""
    y_array = np.asarray(y, dtype=int)
    rows: list[dict[str, Any]] = []
    for outer_fold, outer_rows in inner_assignments.groupby("outer_fold", sort=True):
        outer_train_indices = outer_rows["sample_index"].to_numpy(dtype=int)
        for inner_fold in sorted(outer_rows["inner_fold"].unique()):
            inner_val = outer_rows.loc[
                outer_rows["inner_fold"].eq(inner_fold),
                "sample_index",
            ].to_numpy(dtype=int)
            inner_train = np.setdiff1d(outer_train_indices, inner_val, assume_unique=False)
            rows.append({
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "inner_train_count": len(inner_train),
                "inner_val_count": len(inner_val),
                "inner_train_positive_count": int(y_array[inner_train].sum()),
                "inner_val_positive_count": int(y_array[inner_val].sum()),
                "inner_train_positive_rate": float(y_array[inner_train].mean()),
                "inner_val_positive_rate": float(y_array[inner_val].mean()),
            })
    return pd.DataFrame(rows)


def assert_nested_split_integrity(
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    *,
    n_samples: int,
    outer_splits: int,
    inner_splits: int,
) -> None:
    """Raise if outer/inner split assignments violate leakage-safe structure."""
    if len(outer_assignments) != n_samples:
        raise ValueError("Outer assignments must have exactly one row per sample")
    if outer_assignments["sample_index"].nunique() != n_samples:
        raise ValueError("Each sample must appear exactly once in outer assignments")
    if outer_assignments["outer_fold"].nunique() != outer_splits:
        raise ValueError("Unexpected number of outer folds")

    for outer_fold in sorted(outer_assignments["outer_fold"].unique()):
        outer_val = set(
            outer_assignments.loc[
                outer_assignments["outer_fold"].eq(outer_fold),
                "sample_index",
            ]
        )
        inner_rows = inner_assignments.loc[inner_assignments["outer_fold"].eq(outer_fold)]
        inner_samples = set(inner_rows["sample_index"])
        if outer_val & inner_samples:
            raise ValueError(f"Outer validation samples appear in inner CV for fold {outer_fold}")
        if inner_rows["inner_fold"].nunique() != inner_splits:
            raise ValueError(f"Unexpected number of inner folds for outer fold {outer_fold}")


def build_prescreen_recipe_space(
    methods: Sequence[str] = DEFAULT_NESTED_CV_PRESCREEN_METHODS,
    *,
    include_adaptive_weighted: bool = True,
) -> pd.DataFrame:
    """Build leakage-safe prescreen recipe declarations.

    Weighted recipes are declared as adaptive. Their weights must be estimated inside the
    current training partition in later stages, never from global train labels.
    """
    method_list = list(dict.fromkeys(methods))
    rows: list[dict[str, Any]] = []
    for size in range(1, len(method_list) + 1):
        for combo in combinations(method_list, size):
            rows.append({
                "prescreen_recipe_id": f"ps_mean_{len(rows) + 1:04d}",
                "prescreen_name": prescreen_combo_name(combo, all_count=len(method_list)),
                "prescreen_methods": json.dumps(list(combo)),
                "prescreen_method_count": len(combo),
                "rank_aggregation": "mean_rank_score",
                "adaptive": False,
                "adaptive_method_count": np.nan,
                "leakage_rule": "fit ranking only on the active training partition",
            })
    if include_adaptive_weighted and method_list:
        rows.append({
            "prescreen_recipe_id": "ps_weighted_all_train_only",
            "prescreen_name": "weighted_all__train_only_single_scores",
            "prescreen_methods": json.dumps(method_list),
            "prescreen_method_count": len(method_list),
            "rank_aggregation": "train_only_single_score_weighted",
            "adaptive": True,
            "adaptive_method_count": len(method_list),
            "leakage_rule": "derive weights only from inner-training folds",
        })
        rows.append({
            "prescreen_recipe_id": "ps_weighted_top5_train_only",
            "prescreen_name": "weighted_top5__train_only_single_scores",
            "prescreen_methods": json.dumps(method_list),
            "prescreen_method_count": len(method_list),
            "rank_aggregation": "train_only_single_score_weighted_top_k",
            "adaptive": True,
            "adaptive_method_count": min(5, len(method_list)),
            "leakage_rule": "select and weight methods only from inner-training folds",
        })
    return pd.DataFrame(rows)


def build_feature_size_grid(
    sizes: Sequence[int] = DEFAULT_NESTED_CV_FEATURE_SIZES,
) -> pd.DataFrame:
    """Build top-k feature-size declarations."""
    clean_sizes = sorted({int(size) for size in sizes if int(size) > 0})
    return pd.DataFrame({
        "feature_size_id": [f"k_{size:03d}" for size in clean_sizes],
        "feature_size": clean_sizes,
    })


def build_model_spec_space(
    *,
    include_xgboost: bool = True,
    include_ebm: bool = True,
    include_lambdamart: bool = True,
) -> pd.DataFrame:
    """Build model/HPO declarations for honest inner-CV selection."""
    specs = expanded_hpo_model_specs(
        include_xgboost=include_xgboost,
        include_ebm=include_ebm,
        include_lambdamart=include_lambdamart,
    )
    rows = [
        {
            "model_spec_id": spec.model_name,
            "base_model_family": spec.base_model_family,
            "model_kind": spec.kind,
            "model_params": json.dumps(dict(spec.params), sort_keys=True),
        }
        for spec in specs
    ]
    return pd.DataFrame(rows)


def build_pipeline_recipe_space(
    prescreen_recipes: pd.DataFrame,
    feature_sizes: pd.DataFrame,
    model_specs: pd.DataFrame,
) -> pd.DataFrame:
    """Build the declarative Cartesian recipe space without fitting anything."""
    rows: list[dict[str, Any]] = []
    iterator = product(
        prescreen_recipes["prescreen_recipe_id"],
        feature_sizes["feature_size"],
        model_specs["model_spec_id"],
    )
    model_lookup = model_specs.set_index("model_spec_id")
    prescreen_lookup = prescreen_recipes.set_index("prescreen_recipe_id")
    for idx, (prescreen_id, feature_size, model_spec_id) in enumerate(iterator, start=1):
        model_row = model_lookup.loc[model_spec_id]
        prescreen_row = prescreen_lookup.loc[prescreen_id]
        rows.append({
            "pipeline_recipe_id": f"pipe_{idx:07d}",
            "prescreen_recipe_id": prescreen_id,
            "prescreen_name": prescreen_row["prescreen_name"],
            "feature_size": int(feature_size),
            "model_spec_id": model_spec_id,
            "base_model_family": model_row["base_model_family"],
            "model_kind": model_row["model_kind"],
            "search_stage": "inner_cv_candidate",
        })
    return pd.DataFrame(rows)


def build_recipe_space_summary(
    pipeline_recipes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recipe counts by prescreen size, feature size, and model family."""
    return (
        pipeline_recipes.groupby(["base_model_family", "feature_size"], as_index=False)
        .size()
        .rename(columns={"size": "n_pipeline_recipes"})
        .sort_values(["base_model_family", "feature_size"])
    )


def leakage_contract_table() -> pd.DataFrame:
    """Return the data-access contract for later honest notebooks."""
    return pd.DataFrame([
        {
            "stage": "outer_fold_creation",
            "allowed_data": "y_train only for stratified splitting",
            "forbidden_data": "no feature ranking, model choice, or score optimization",
        },
        {
            "stage": "inner_prescreen_fit",
            "allowed_data": "X_inner_train and y_inner_train",
            "forbidden_data": "X_inner_val, y_inner_val, X_outer_val, y_outer_val",
        },
        {
            "stage": "inner_recipe_scoring",
            "allowed_data": "predictions on inner_val and y_inner_val",
            "forbidden_data": "outer_val labels or global train-wide prescreen rankings",
        },
        {
            "stage": "outer_refit",
            "allowed_data": (
                "X_outer_train and y_outer_train plus recipe selected inside outer_train"
            ),
            "forbidden_data": "X_outer_val and y_outer_val during selection or fitting",
        },
        {
            "stage": "outer_evaluation",
            "allowed_data": (
                "predictions on X_outer_val and y_outer_val for final fold scoring only"
            ),
            "forbidden_data": "using outer_val score to alter that fold's selected recipe",
        },
        {
            "stage": "final_test_ranking",
            "allowed_data": "all X_train/y_train for final recipe refit and X_test for scoring",
            "forbidden_data": "test labels, which are not available",
        },
    ])


def empty_experiment_log() -> pd.DataFrame:
    """Return an empty experiment log with stable columns."""
    return pd.DataFrame(columns=EXPERIMENT_LOG_COLUMNS)


def write_stage_one_outputs(
    *,
    output_dir: Path,
    outer_assignments: pd.DataFrame,
    outer_summary: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    inner_summary: pd.DataFrame,
    prescreen_recipes: pd.DataFrame,
    feature_sizes: pd.DataFrame,
    model_specs: pd.DataFrame,
    pipeline_recipes: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Persist all stage-one manifests and logs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_assignments.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    outer_summary.to_csv(output_dir / "outer_fold_summary.csv", index=False)
    inner_assignments.to_csv(output_dir / "inner_fold_assignments.csv", index=False)
    inner_summary.to_csv(output_dir / "inner_fold_summary.csv", index=False)
    prescreen_recipes.to_csv(output_dir / "prescreen_recipe_space.csv", index=False)
    feature_sizes.to_csv(output_dir / "feature_size_grid.csv", index=False)
    model_specs.to_csv(output_dir / "model_spec_space.csv", index=False)
    pipeline_recipes.to_csv(output_dir / "pipeline_recipe_space.csv", index=False)
    build_recipe_space_summary(pipeline_recipes).to_csv(
        output_dir / "pipeline_recipe_space_summary.csv",
        index=False,
    )
    leakage_contract_table().to_csv(output_dir / "leakage_contract.csv", index=False)
    empty_experiment_log().to_csv(output_dir / "experiment_log.csv", index=False)
    with (output_dir / "stage_config.json").open("w") as file_obj:
        json.dump(config, file_obj, indent=2)
