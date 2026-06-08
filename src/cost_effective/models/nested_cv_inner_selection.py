"""Leakage-safe inner-CV recipe selection helpers."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - sklearn normally provides this dependency.
    threadpool_limits = None

from cost_effective.dataset.ensemble_feature_selection import (
    collect_prescreen_rankings,
    combine_prescreen_rankings,
    combine_prescreen_rankings_weighted,
)
from cost_effective.dataset.utils import DEFAULT_MAX_TARGETS
from cost_effective.models.business_scoring import feature_cost_scale, scaled_business_curve
from cost_effective.models.model_family_validation import (
    feature_set_key,
    fit_lambdamart_full,
    make_classifier_spec,
    positive_scores,
)
from cost_effective.models.modeling import build_f1_curve
from cost_effective.models.nested_cv_inner_scoring import (
    _inner_score_row,
    aggregate_inner_scores,
    append_csv,
    filter_model_specs,
    filter_prescreen_recipes,
    has_inner_prediction_columns,
    score_inner_oof_baselines,
    score_inner_oof_prediction_blocks,
    score_inner_oof_predictions,
    score_inner_oof_predictions_csv,
    select_top_feature_recipes,
    select_top_pipeline_recipes,
)
from cost_effective.models.parsing import (
    parse_json_dict,
    parse_json_list,
)

__all__ = [
    "aggregate_inner_scores",
    "append_csv",
    "build_fold_feature_candidates",
    "evaluate_feature_candidates",
    "evaluate_feature_candidates_oof_blocks",
    "evaluate_feature_candidates_predictions",
    "feature_cost_scale",
    "filter_model_specs",
    "filter_prescreen_recipes",
    "fit_predict_model_spec",
    "has_inner_prediction_columns",
    "inner_fold_indices",
    "load_stage_one_tables",
    "materialize_prescreen_ranking",
    "outer_fold_indices",
    "parse_json_dict",
    "parse_json_list",
    "required_prescreen_methods",
    "restrict_feature_candidates_to_selected_pairs",
    "run_inner_fold_round",
    "run_inner_fold_round_oof_blocks",
    "run_inner_fold_round_predictions",
    "score_inner_oof_baselines",
    "score_inner_oof_prediction_blocks",
    "score_inner_oof_predictions",
    "score_inner_oof_predictions_csv",
    "score_validation_predictions_scaled",
    "select_top_feature_recipes",
    "select_top_pipeline_recipes",
    "train_only_prescreen_weights",
]


def load_stage_one_tables(stage_one_dir: Path) -> dict[str, pd.DataFrame]:
    """Load fold and recipe-space artifacts from stage 1."""
    return {
        "outer_assignments": pd.read_csv(stage_one_dir / "outer_fold_assignments.csv"),
        "inner_assignments": pd.read_csv(stage_one_dir / "inner_fold_assignments.csv"),
        "prescreen_recipes": pd.read_csv(stage_one_dir / "prescreen_recipe_space.csv"),
        "feature_sizes": pd.read_csv(stage_one_dir / "feature_size_grid.csv"),
        "model_specs": pd.read_csv(stage_one_dir / "model_spec_space.csv"),
        "pipeline_recipes": pd.read_csv(stage_one_dir / "pipeline_recipe_space.csv"),
        "leakage_contract": pd.read_csv(stage_one_dir / "leakage_contract.csv"),
    }


def outer_fold_indices(
    outer_assignments: pd.DataFrame,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outer-train and outer-validation sample indices."""
    outer_fold = int(outer_fold)
    val_idx = outer_assignments.loc[
        outer_assignments["outer_fold"].eq(outer_fold),
        "sample_index",
    ].to_numpy(dtype=int)
    train_idx = outer_assignments.loc[
        ~outer_assignments["outer_fold"].eq(outer_fold),
        "sample_index",
    ].to_numpy(dtype=int)
    return train_idx, val_idx


def inner_fold_indices(
    inner_assignments: pd.DataFrame,
    outer_fold: int,
    inner_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inner-train and inner-validation indices inside an outer fold."""
    outer_rows = inner_assignments.loc[inner_assignments["outer_fold"].eq(int(outer_fold))]
    val_idx = outer_rows.loc[
        outer_rows["inner_fold"].eq(int(inner_fold)),
        "sample_index",
    ].to_numpy(dtype=int)
    all_outer_train = outer_rows["sample_index"].to_numpy(dtype=int)
    train_idx = np.setdiff1d(all_outer_train, val_idx, assume_unique=False)
    return train_idx, val_idx


def required_prescreen_methods(prescreen_recipes: pd.DataFrame) -> list[str]:
    """Return unique prescreen methods needed by a recipe subset."""
    methods: list[str] = []
    for value in prescreen_recipes["prescreen_methods"]:
        methods.extend(parse_json_list(value))
    return list(dict.fromkeys(methods))


def train_only_prescreen_weights(
    rankings: pd.DataFrame,
    methods: Sequence[str],
    *,
    top_n: int = 25,
    min_weight: float = 0.05,
) -> dict[str, float]:
    """Derive safe adaptive weights from train-only rank-score concentration.

    This does not use validation labels. It only inspects rankings fitted on the active
    training partition and rewards methods with sharper top-ranked features.
    """
    rows: list[tuple[str, float]] = []
    for method in methods:
        method_scores = rankings.loc[rankings["method"].eq(method), "rank_score"]
        if method_scores.empty:
            continue
        top_score = float(method_scores.sort_values(ascending=False).head(top_n).mean())
        rows.append((method, top_score))
    if not rows:
        return {}
    values = np.asarray([score for _method, score in rows], dtype=float)
    shifted = values - float(values.min())
    if float(shifted.sum()) <= 0:
        shifted = np.ones_like(values)
    shifted = shifted + min_weight * max(float(shifted.mean()), 1.0)
    weights = shifted / shifted.sum()
    return {method: float(weight) for (method, _score), weight in zip(rows, weights, strict=True)}


def materialize_prescreen_ranking(
    rankings: pd.DataFrame,
    recipe: pd.Series,
) -> pd.DataFrame | None:
    """Materialize a fold-local ranking for one prescreen recipe."""
    methods = parse_json_list(recipe["prescreen_methods"])
    present = set(rankings["method"].drop_duplicates())
    if not methods or any(method not in present for method in methods):
        return None

    aggregation = str(recipe.get("rank_aggregation", "mean_rank_score"))
    name = str(recipe["prescreen_name"])
    if aggregation == "mean_rank_score":
        return combine_prescreen_rankings(rankings, methods, name=name)

    weights = train_only_prescreen_weights(rankings, methods)
    if not weights:
        return None
    if aggregation == "train_only_single_score_weighted_top_k":
        top_k = int(recipe.get("adaptive_method_count", 5) or 5)
        weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return combine_prescreen_rankings_weighted(rankings, weights, name=name)


def build_fold_feature_candidates(
    rankings: pd.DataFrame,
    prescreen_recipes: pd.DataFrame,
    feature_sizes: Sequence[int],
    *,
    max_feature_candidates: int | None = None,
    dedupe_feature_sets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top-k feature candidates from fold-local prescreen rankings."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    clean_sizes = sorted({int(size) for size in feature_sizes if int(size) > 0})

    for _, recipe in prescreen_recipes.iterrows():
        ranking = materialize_prescreen_ranking(rankings, recipe)
        if ranking is None or ranking.empty:
            failures.append({
                "prescreen_recipe_id": recipe.get("prescreen_recipe_id", ""),
                "prescreen_name": recipe.get("prescreen_name", ""),
                "status": "skipped",
                "error": "required rankings unavailable",
            })
            continue
        ordered = ranking.sort_values("order")["feature"].tolist()
        score_lookup = ranking.set_index("feature")["ensemble_score"]
        for size in clean_sizes:
            if size > len(ordered):
                continue
            features = ordered[:size]
            key = feature_set_key(features)
            if not key:
                continue
            dedupe_key = key if dedupe_feature_sets else (*key, str(recipe["prescreen_recipe_id"]))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            heuristic = float(score_lookup.loc[list(key)].mean())
            rows.append({
                "feature_recipe_id": (f"{recipe['prescreen_recipe_id']}__k_{int(size):03d}"),
                "prescreen_recipe_id": recipe["prescreen_recipe_id"],
                "prescreen_name": recipe["prescreen_name"],
                "prescreen_methods": recipe["prescreen_methods"],
                "rank_aggregation": recipe["rank_aggregation"],
                "feature_size": int(size),
                "selected_features": list(key),
                "selected_features_text": ",".join(key),
                "feature_key": "|".join(key),
                "candidate_heuristic_score": heuristic,
            })

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["candidate_heuristic_score", "feature_size", "feature_recipe_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        if max_feature_candidates is not None:
            candidates = candidates.head(max_feature_candidates).copy()
    return candidates, pd.DataFrame(failures)


def restrict_feature_candidates_to_selected_pairs(
    feature_candidates: pd.DataFrame,
    selected_feature_recipes: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only exact prescreen/feature-size pairs selected by round 1."""
    if feature_candidates.empty or selected_feature_recipes.empty:
        return feature_candidates.head(0).copy()
    selected_pairs = {
        (str(row["prescreen_recipe_id"]), int(row["feature_size"]))
        for _, row in selected_feature_recipes.iterrows()
    }
    mask = feature_candidates.apply(
        lambda row: (str(row["prescreen_recipe_id"]), int(row["feature_size"])) in selected_pairs,
        axis=1,
    )
    return feature_candidates.loc[mask].reset_index(drop=True)


def fit_predict_model_spec(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    features: Sequence[str],
    model_spec: pd.Series,
    *,
    random_state: int,
) -> np.ndarray:
    """Fit one model spec on the active train partition and predict validation scores."""
    feature_list = list(features)
    return _fit_predict_model_spec_on_feature_frame(
        X_train[feature_list],
        y_train,
        X_val[feature_list],
        model_spec,
        random_state=random_state,
    )


def _fit_predict_model_spec_on_feature_frame(
    X_train_features: pd.DataFrame,
    y_train: pd.Series,
    X_val_features: pd.DataFrame,
    model_spec: pd.Series,
    *,
    random_state: int,
) -> np.ndarray:
    """Fit one model spec using already materialized feature columns."""
    base_family = str(model_spec["base_model_family"])
    kind = str(model_spec["model_kind"])
    params = parse_json_dict(model_spec["model_params"])
    if kind == "classifier":
        spec = make_classifier_spec(base_family, params, random_state=random_state)
        if spec.factory is None:
            raise ValueError(f"Missing classifier factory for {base_family}")
        model = spec.factory(y_train.to_numpy())
        model.fit(X_train_features, y_train)
        return positive_scores(model, X_val_features)
    if kind == "lambdamart":
        model = fit_lambdamart_full(
            X_train_features,
            y_train,
            list(X_train_features.columns),
            params=params,
            random_state=random_state,
        )
        return np.asarray(model.predict(X_val_features), dtype=float)
    raise ValueError(f"Unknown model kind: {kind}")


def score_validation_predictions_scaled(
    y: pd.Series,
    validation_scores: np.ndarray,
    source_feature_names: Sequence[str],
    *,
    method: str,
    candidate_column_count: int | None = None,
    max_targets: int = DEFAULT_MAX_TARGETS,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
) -> dict[str, Any]:
    """Score validation rankings with a row-scaled feature-acquisition penalty."""
    unique_sources = tuple(dict.fromkeys(source_feature_names))
    feature_count = len(unique_sources)
    score_array = np.asarray(validation_scores, dtype=float)
    curve = scaled_business_curve(
        y,
        score_array,
        feature_count=feature_count,
        max_targets=max_targets,
        feature_cost_reference_row_count=feature_cost_reference_row_count,
    )
    best_row = curve.iloc[int(curve["business_score"].idxmax())]

    try:
        f1 = build_f1_curve(
            pd.Series(np.asarray(y, dtype=int)), score_array, max_targets=max_targets
        )
        f1_score = f1.best_f1
        roc_auc_score = f1.roc_auc
    except ValueError:
        f1_score = np.nan
        roc_auc_score = np.nan

    return {
        "method": method,
        "candidate_column_count": candidate_column_count or feature_count,
        "source_feature_count": feature_count,
        "source_features": list(unique_sources),
        "scoring_version": "scaled_full_objective_v1",
        "feature_cost_reference_row_count": int(feature_cost_reference_row_count),
        "evaluation_row_count": int(best_row["evaluation_row_count"]),
        "target_limit": int(best_row["target_limit"]),
        "feature_cost_scale": float(best_row["feature_cost_scale"]),
        "scaled_true_positive_value": float(best_row["scaled_true_positive_value"]),
        "scaled_false_positive_cost": float(best_row["scaled_false_positive_cost"]),
        "scaled_feature_cost": float(best_row["scaled_feature_cost"]),
        "unscaled_feature_cost": float(best_row["unscaled_feature_cost"]),
        "oof_business_score": float(best_row["business_score"]),
        "oof_business_score_unscaled_feature_cost": float(
            best_row["business_score_unscaled_feature_cost"]
        ),
        "oof_business_value_before_feature_cost": float(
            best_row["business_value_before_feature_cost"]
        ),
        "oof_optimal_k": int(best_row["k"]),
        "oof_tp_at_optimal_k": int(best_row["tp"]),
        "oof_fp_at_optimal_k": int(best_row["fp"]),
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
    }


def _native_thread_limit(max_threads: int = 1):
    """Limit native BLAS/OpenMP pools while joblib fans out Python-level fits."""
    if threadpool_limits is None or int(max_threads) <= 0:
        return nullcontext()
    return threadpool_limits(limits=int(max_threads))


def _evaluate_candidate_model_from_feature_frame(
    X_train_features: pd.DataFrame,
    y_train: pd.Series,
    X_val_features: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_spec: pd.Series,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_targets: int,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
) -> dict[str, Any]:
    features = list(X_train_features.columns)
    pipeline_recipe_id = f"{candidate['feature_recipe_id']}__{model_spec['model_spec_id']}"
    method = f"outer{outer_fold:02d}__inner{inner_fold:02d}__{round_name}__{pipeline_recipe_id}"
    started = time.perf_counter()
    try:
        scores = _fit_predict_model_spec_on_feature_frame(
            X_train_features,
            y_train,
            X_val_features,
            model_spec,
            random_state=random_state + outer_fold * 10000 + inner_fold * 100,
        )
        scored = score_validation_predictions_scaled(
            y_val,
            scores,
            features,
            method=method,
            candidate_column_count=len(features),
            max_targets=max_targets,
            feature_cost_reference_row_count=feature_cost_reference_row_count,
        )
        return _inner_score_row(
            candidate,
            model_spec,
            scored,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            round_name=round_name,
            pipeline_recipe_id=pipeline_recipe_id,
            status="ok",
            error="",
            duration_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return _inner_score_row(
            candidate,
            model_spec,
            {"source_features": features},
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            round_name=round_name,
            pipeline_recipe_id=pipeline_recipe_id,
            status="error",
            error=repr(exc),
            duration_seconds=time.perf_counter() - started,
        )


def _inner_prediction_base_row(
    candidate: pd.Series,
    model_spec: pd.Series,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    pipeline_recipe_id: str,
    selected_features: Sequence[str],
) -> dict[str, Any]:
    features = [str(feature) for feature in selected_features]
    return {
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "round_name": round_name,
        "pipeline_recipe_id": pipeline_recipe_id,
        "feature_recipe_id": candidate.get("feature_recipe_id", ""),
        "prescreen_recipe_id": candidate.get("prescreen_recipe_id", ""),
        "prescreen_name": candidate.get("prescreen_name", ""),
        "prescreen_methods": candidate.get("prescreen_methods", ""),
        "rank_aggregation": candidate.get("rank_aggregation", ""),
        "feature_size": int(candidate.get("feature_size", len(features))),
        "selected_features": features,
        "selected_features_text": ",".join(features),
        "candidate_heuristic_score": candidate.get("candidate_heuristic_score", np.nan),
        "model_spec_id": model_spec.get("model_spec_id", ""),
        "base_model_family": model_spec.get("base_model_family", ""),
        "model_kind": model_spec.get("model_kind", ""),
        "model_params": model_spec.get("model_params", ""),
    }


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate_candidate_model_predictions_from_feature_frame(
    X_train_features: pd.DataFrame,
    y_train: pd.Series,
    X_val_features: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_spec: pd.Series,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
) -> list[dict[str, Any]]:
    features = list(X_train_features.columns)
    pipeline_recipe_id = f"{candidate['feature_recipe_id']}__{model_spec['model_spec_id']}"
    started = time.perf_counter()
    base = _inner_prediction_base_row(
        candidate,
        model_spec,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        round_name=round_name,
        pipeline_recipe_id=pipeline_recipe_id,
        selected_features=features,
    )
    try:
        scores = _fit_predict_model_spec_on_feature_frame(
            X_train_features,
            y_train,
            X_val_features,
            model_spec,
            random_state=random_state + outer_fold * 10000 + inner_fold * 100,
        )
        score_array = np.asarray(scores, dtype=float)
        if len(score_array) != len(y_val):
            raise ValueError("model returned a score vector with unexpected length")
        duration_seconds = time.perf_counter() - started
        sample_indices = list(y_val.index)
        y_values = np.asarray(y_val, dtype=int)
        rows: list[dict[str, Any]] = []
        for sample_index, y_true, score in zip(sample_indices, y_values, score_array, strict=True):
            rows.append({
                **base,
                "sample_index": _python_scalar(sample_index),
                "y_true": int(y_true),
                "score": float(score),
                "status": "ok",
                "error": "",
                "duration_seconds": float(duration_seconds),
            })
        return rows
    except Exception as exc:
        return [
            {
                **base,
                "sample_index": np.nan,
                "y_true": np.nan,
                "score": np.nan,
                "status": "error",
                "error": repr(exc),
                "duration_seconds": float(time.perf_counter() - started),
            }
        ]


def _evaluate_candidate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_spec: pd.Series,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_targets: int,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
) -> dict[str, Any]:
    features = list(candidate["selected_features"])
    return _evaluate_candidate_model_from_feature_frame(
        X_train[features],
        y_train,
        X_val[features],
        y_val,
        candidate,
        model_spec,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        round_name=round_name,
        random_state=random_state,
        max_targets=max_targets,
        feature_cost_reference_row_count=feature_cost_reference_row_count,
    )


def _evaluate_candidate_model_group(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_specs: Sequence[pd.Series],
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_targets: int,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
) -> list[dict[str, Any]]:
    features = list(candidate["selected_features"])
    x_train_features = X_train[features]
    x_val_features = X_val[features]
    return [
        _evaluate_candidate_model_from_feature_frame(
            x_train_features,
            y_train,
            x_val_features,
            y_val,
            candidate,
            model_spec,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            round_name=round_name,
            random_state=random_state,
            max_targets=max_targets,
            feature_cost_reference_row_count=feature_cost_reference_row_count,
        )
        for model_spec in model_specs
    ]


def _evaluate_candidate_model_prediction_group(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_specs: Sequence[pd.Series],
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
) -> list[dict[str, Any]]:
    features = list(candidate["selected_features"])
    x_train_features = X_train[features]
    x_val_features = X_val[features]
    rows: list[dict[str, Any]] = []
    for model_spec in model_specs:
        rows.extend(
            _evaluate_candidate_model_predictions_from_feature_frame(
                x_train_features,
                y_train,
                x_val_features,
                y_val,
                candidate,
                model_spec,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                round_name=round_name,
                random_state=random_state,
            )
        )
    return rows


def evaluate_feature_candidates_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_candidates: pd.DataFrame,
    model_specs: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    log_every: int = 100,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    parallel_verbose: int = 0,
) -> pd.DataFrame:
    """Fit fold-local candidates and return inner-validation predictions.

    Prescreening and candidate materialization happen before this helper on the active
    inner-train partition. This function only fits each candidate/model on
    ``X_train``/``y_train`` and emits scores for ``X_val`` rows; it does not use
    validation labels except to carry ``y_true`` into the later OOF scorer.
    """
    candidates = [candidate for _, candidate in feature_candidates.iterrows()]
    model_spec_rows = [model_spec for _, model_spec in model_specs.iterrows()]
    total = len(candidates) * len(model_spec_rows)
    if total == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    completed = 0
    next_log_at = int(log_every) if int(log_every) > 0 else 0

    def log_progress() -> None:
        nonlocal next_log_at
        if next_log_at <= 0:
            return
        while completed >= next_log_at:
            print(
                f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
                f"{completed}/{total} model fits completed"
            )
            next_log_at += int(log_every)

    if n_jobs == 1:
        with _native_thread_limit(1):
            for candidate in candidates:
                group_rows = _evaluate_candidate_model_prediction_group(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    candidate,
                    model_spec_rows,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    round_name=round_name,
                    random_state=random_state,
                )
                rows.extend(group_rows)
                completed += len(model_spec_rows)
                log_progress()
        return pd.DataFrame(rows)

    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"starting {total} model fits as {len(candidates)} candidate tasks "
        f"with n_jobs={n_jobs}, prefer={parallel_prefer}"
    )
    with _native_thread_limit(1):
        row_groups = Parallel(
            n_jobs=n_jobs,
            prefer=parallel_prefer,
            verbose=parallel_verbose,
            return_as="generator_unordered",
            batch_size=1,
        )(
            delayed(_evaluate_candidate_model_prediction_group)(
                X_train,
                y_train,
                X_val,
                y_val,
                candidate,
                model_spec_rows,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                round_name=round_name,
                random_state=random_state,
            )
            for candidate in candidates
        )
        for group_rows in row_groups:
            rows.extend(group_rows)
            completed += len(model_spec_rows)
            log_progress()
    print(f"[outer={outer_fold} inner={inner_fold} round={round_name}] finished {total} model fits")
    return pd.DataFrame(rows)


def evaluate_feature_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_candidates: pd.DataFrame,
    model_specs: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_targets: int = DEFAULT_MAX_TARGETS,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
    log_every: int = 100,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    parallel_verbose: int = 0,
) -> pd.DataFrame:
    """Evaluate fold-local feature candidates with model specs on inner validation."""
    candidates = [candidate for _, candidate in feature_candidates.iterrows()]
    model_spec_rows = [model_spec for _, model_spec in model_specs.iterrows()]
    total = len(candidates) * len(model_spec_rows)
    if total == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    completed = 0
    next_log_at = int(log_every) if int(log_every) > 0 else 0

    def log_progress() -> None:
        nonlocal next_log_at
        if next_log_at <= 0:
            return
        while completed >= next_log_at:
            print(
                f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
                f"{completed}/{total} model fits completed"
            )
            next_log_at += int(log_every)

    if n_jobs == 1:
        with _native_thread_limit(1):
            for candidate in candidates:
                group_rows = _evaluate_candidate_model_group(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    candidate,
                    model_spec_rows,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    round_name=round_name,
                    random_state=random_state,
                    max_targets=max_targets,
                    feature_cost_reference_row_count=feature_cost_reference_row_count,
                )
                rows.extend(group_rows)
                completed += len(group_rows)
                log_progress()
        return pd.DataFrame(rows)

    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"starting {total} model fits as {len(candidates)} candidate tasks "
        f"with n_jobs={n_jobs}, prefer={parallel_prefer}"
    )
    with _native_thread_limit(1):
        row_groups = Parallel(
            n_jobs=n_jobs,
            prefer=parallel_prefer,
            verbose=parallel_verbose,
            return_as="generator_unordered",
            batch_size=1,
        )(
            delayed(_evaluate_candidate_model_group)(
                X_train,
                y_train,
                X_val,
                y_val,
                candidate,
                model_spec_rows,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                round_name=round_name,
                random_state=random_state,
                max_targets=max_targets,
                feature_cost_reference_row_count=feature_cost_reference_row_count,
            )
            for candidate in candidates
        )
        for group_rows in row_groups:
            rows.extend(group_rows)
            completed += len(group_rows)
            log_progress()
    print(f"[outer={outer_fold} inner={inner_fold} round={round_name}] finished {total} model fits")
    return pd.DataFrame(rows)


def run_inner_fold_round(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    prescreen_recipes: pd.DataFrame,
    feature_sizes: Sequence[int],
    model_specs: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_feature_candidates: int | None,
    max_targets: int = DEFAULT_MAX_TARGETS,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
    log_every: int = 100,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    parallel_verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one leakage-safe inner fold round."""
    x_inner_train = X.iloc[list(train_idx)]
    y_inner_train = y.iloc[list(train_idx)]
    x_inner_val = X.iloc[list(val_idx)]
    y_inner_val = y.iloc[list(val_idx)]

    methods = required_prescreen_methods(prescreen_recipes)
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"fitting {len(methods)} prescreen methods on {len(train_idx)} inner-train rows"
    )
    rankings, method_failures = collect_prescreen_rankings(
        x_inner_train,
        y_inner_train,
        methods=methods,
        random_state=random_state + outer_fold * 10000 + inner_fold * 100,
        continue_on_error=True,
    )
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"available methods={rankings['method'].nunique() if not rankings.empty else 0}; "
        f"failures={len(method_failures)}"
    )

    feature_candidates, recipe_failures = build_fold_feature_candidates(
        rankings,
        prescreen_recipes,
        feature_sizes,
        max_feature_candidates=max_feature_candidates,
    )
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"feature candidates={len(feature_candidates)}; model specs={len(model_specs)}"
    )
    if feature_candidates.empty or model_specs.empty:
        return pd.DataFrame(), method_failures, recipe_failures

    scores = evaluate_feature_candidates(
        x_inner_train,
        y_inner_train,
        x_inner_val,
        y_inner_val,
        feature_candidates,
        model_specs,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        round_name=round_name,
        random_state=random_state,
        max_targets=max_targets,
        feature_cost_reference_row_count=feature_cost_reference_row_count,
        log_every=log_every,
        n_jobs=n_jobs,
        parallel_prefer=parallel_prefer,
        parallel_verbose=parallel_verbose,
    )
    return scores, method_failures, recipe_failures


def run_inner_fold_round_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    prescreen_recipes: pd.DataFrame,
    feature_sizes: Sequence[int],
    model_specs: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_feature_candidates: int | None,
    log_every: int = 100,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    parallel_verbose: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one leakage-safe inner fold and return validation predictions."""
    x_inner_train = X.iloc[list(train_idx)]
    y_inner_train = y.iloc[list(train_idx)]
    x_inner_val = X.iloc[list(val_idx)]
    y_inner_val = y.iloc[list(val_idx)]

    methods = required_prescreen_methods(prescreen_recipes)
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"fitting {len(methods)} prescreen methods on {len(train_idx)} inner-train rows"
    )
    rankings, method_failures = collect_prescreen_rankings(
        x_inner_train,
        y_inner_train,
        methods=methods,
        random_state=random_state + outer_fold * 10000 + inner_fold * 100,
        continue_on_error=True,
    )
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"available methods={rankings['method'].nunique() if not rankings.empty else 0}; "
        f"failures={len(method_failures)}"
    )

    feature_candidates, recipe_failures = build_fold_feature_candidates(
        rankings,
        prescreen_recipes,
        feature_sizes,
        max_feature_candidates=max_feature_candidates,
    )
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"feature candidates={len(feature_candidates)}; model specs={len(model_specs)}"
    )
    if feature_candidates.empty or model_specs.empty:
        return pd.DataFrame(), method_failures, recipe_failures

    predictions = evaluate_feature_candidates_predictions(
        x_inner_train,
        y_inner_train,
        x_inner_val,
        y_inner_val,
        feature_candidates,
        model_specs,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        round_name=round_name,
        random_state=random_state,
        log_every=log_every,
        n_jobs=n_jobs,
        parallel_prefer=parallel_prefer,
        parallel_verbose=parallel_verbose,
    )
    return predictions, method_failures, recipe_failures


def _evaluate_candidate_model_oof_block_from_feature_frame(
    X_train_features: pd.DataFrame,
    y_train: pd.Series,
    X_val_features: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_spec: pd.Series,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
) -> dict[str, Any]:
    """Fit one model and return compact validation arrays for later OOF scoring."""
    features = list(X_train_features.columns)
    pipeline_recipe_id = f"{candidate['feature_recipe_id']}__{model_spec['model_spec_id']}"
    started = time.perf_counter()
    base = _inner_prediction_base_row(
        candidate,
        model_spec,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        round_name=round_name,
        pipeline_recipe_id=pipeline_recipe_id,
        selected_features=features,
    )
    try:
        scores = _fit_predict_model_spec_on_feature_frame(
            X_train_features,
            y_train,
            X_val_features,
            model_spec,
            random_state=random_state + outer_fold * 10000 + inner_fold * 100,
        )
        score_array = np.asarray(scores, dtype=float)
        if len(score_array) != len(y_val):
            raise ValueError("model returned a score vector with unexpected length")
        return {
            **base,
            "sample_index": np.asarray(list(y_val.index)),
            "y_true": np.asarray(y_val, dtype=int),
            "score": score_array,
            "status": "ok",
            "error": "",
            "duration_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            **base,
            "sample_index": np.asarray([], dtype=int),
            "y_true": np.asarray([], dtype=int),
            "score": np.asarray([], dtype=float),
            "status": "error",
            "error": repr(exc),
            "duration_seconds": float(time.perf_counter() - started),
        }


def _evaluate_candidate_model_oof_block_group(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: pd.Series,
    model_specs: Sequence[pd.Series],
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
) -> list[dict[str, Any]]:
    features = list(candidate["selected_features"])
    x_train_features = X_train[features]
    x_val_features = X_val[features]
    return [
        _evaluate_candidate_model_oof_block_from_feature_frame(
            x_train_features,
            y_train,
            x_val_features,
            y_val,
            candidate,
            model_spec,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            round_name=round_name,
            random_state=random_state,
        )
        for model_spec in model_specs
    ]


def evaluate_feature_candidates_oof_blocks(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_candidates: pd.DataFrame,
    model_specs: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    log_every: int = 100,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    parallel_verbose: int = 0,
) -> list[dict[str, Any]]:
    """Fit fold-local candidates and return compact arrays, not sample-level rows."""
    candidates = [candidate for _, candidate in feature_candidates.iterrows()]
    model_spec_rows = [model_spec for _, model_spec in model_specs.iterrows()]
    total = len(candidates) * len(model_spec_rows)
    if total == 0:
        return []

    blocks: list[dict[str, Any]] = []
    completed = 0
    next_log_at = int(log_every) if int(log_every) > 0 else 0

    def log_progress() -> None:
        nonlocal next_log_at
        if next_log_at <= 0:
            return
        while completed >= next_log_at:
            print(
                f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
                f"{completed}/{total} model fits completed"
            )
            next_log_at += int(log_every)

    if n_jobs == 1:
        with _native_thread_limit(1):
            for candidate in candidates:
                group_blocks = _evaluate_candidate_model_oof_block_group(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    candidate,
                    model_spec_rows,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    round_name=round_name,
                    random_state=random_state,
                )
                blocks.extend(group_blocks)
                completed += len(model_spec_rows)
                log_progress()
        return blocks

    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"starting {total} model fits as {len(candidates)} candidate tasks "
        f"with n_jobs={n_jobs}, prefer={parallel_prefer}"
    )
    with _native_thread_limit(1):
        block_groups = Parallel(
            n_jobs=n_jobs,
            prefer=parallel_prefer,
            verbose=parallel_verbose,
            return_as="generator_unordered",
            batch_size=1,
        )(
            delayed(_evaluate_candidate_model_oof_block_group)(
                X_train,
                y_train,
                X_val,
                y_val,
                candidate,
                model_spec_rows,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                round_name=round_name,
                random_state=random_state,
            )
            for candidate in candidates
        )
        for group_blocks in block_groups:
            blocks.extend(group_blocks)
            completed += len(model_spec_rows)
            log_progress()
    print(f"[outer={outer_fold} inner={inner_fold} round={round_name}] finished {total} model fits")
    return blocks


def run_inner_fold_round_oof_blocks(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    prescreen_recipes: pd.DataFrame,
    feature_sizes: Sequence[int],
    model_specs: pd.DataFrame,
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    random_state: int,
    max_feature_candidates: int | None,
    log_every: int = 100,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    parallel_verbose: int = 0,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Run one leakage-safe inner fold and keep validation scores in compact blocks."""
    x_inner_train = X.iloc[list(train_idx)]
    y_inner_train = y.iloc[list(train_idx)]
    x_inner_val = X.iloc[list(val_idx)]
    y_inner_val = y.iloc[list(val_idx)]

    methods = required_prescreen_methods(prescreen_recipes)
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"fitting {len(methods)} prescreen methods on {len(train_idx)} inner-train rows"
    )
    rankings, method_failures = collect_prescreen_rankings(
        x_inner_train,
        y_inner_train,
        methods=methods,
        random_state=random_state + outer_fold * 10000 + inner_fold * 100,
        continue_on_error=True,
    )
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"available methods={rankings['method'].nunique() if not rankings.empty else 0}; "
        f"failures={len(method_failures)}"
    )

    feature_candidates, recipe_failures = build_fold_feature_candidates(
        rankings,
        prescreen_recipes,
        feature_sizes,
        max_feature_candidates=max_feature_candidates,
    )
    print(
        f"[outer={outer_fold} inner={inner_fold} round={round_name}] "
        f"feature candidates={len(feature_candidates)}; model specs={len(model_specs)}"
    )
    if feature_candidates.empty or model_specs.empty:
        return [], method_failures, recipe_failures

    blocks = evaluate_feature_candidates_oof_blocks(
        x_inner_train,
        y_inner_train,
        x_inner_val,
        y_inner_val,
        feature_candidates,
        model_specs,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        round_name=round_name,
        random_state=random_state,
        log_every=log_every,
        n_jobs=n_jobs,
        parallel_prefer=parallel_prefer,
        parallel_verbose=parallel_verbose,
    )
    return blocks, method_failures, recipe_failures
