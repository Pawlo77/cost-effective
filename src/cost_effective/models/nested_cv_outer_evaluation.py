"""Honest outer-fold evaluation and final refit helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cost_effective.dataset.ensemble_feature_selection import collect_prescreen_rankings
from cost_effective.dataset.utils import DEFAULT_MAX_TARGETS
from cost_effective.models.business_scoring import topk_business_curve
from cost_effective.models.modeling import build_f1_curve
from cost_effective.models.nested_cv_inner_selection import (
    build_fold_feature_candidates,
    fit_predict_model_spec,
    has_inner_prediction_columns,
    outer_fold_indices,
    required_prescreen_methods,
    score_inner_oof_predictions_csv,
    select_top_pipeline_recipes,
)
from cost_effective.models.parsing import parse_feature_names


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def recipe_config_id(row: Mapping[str, Any] | pd.Series) -> str:
    """Stable id for a prescreen/feature-size/model configuration."""
    feature_size = int(float(row["feature_size"]))
    return f"{row['prescreen_recipe_id']}__k_{feature_size:03d}__{row['model_spec_id']}"


def _prescreen_id_from_recipe_fields(row: Mapping[str, Any] | pd.Series) -> str:
    for column in ("feature_recipe_id", "pipeline_recipe_id"):
        value = row.get(column, "")
        if isinstance(value, str) and "__k_" in value:
            return value.split("__k_", maxsplit=1)[0]
    return ""


def normalize_pipeline_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Clean selected pipeline recipes loaded from stage 02 artifacts."""
    if recipes.empty:
        return recipes.copy()
    out = recipes.copy()
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="coerce")
    out = out.loc[out["outer_fold"].notna()].copy()
    out["outer_fold"] = out["outer_fold"].astype(int)
    out["feature_size"] = pd.to_numeric(out["feature_size"], errors="coerce").astype(int)
    if "prescreen_recipe_id" not in out.columns:
        out["prescreen_recipe_id"] = out.apply(_prescreen_id_from_recipe_fields, axis=1)
    else:
        missing_prescreen = out["prescreen_recipe_id"].isna() | out["prescreen_recipe_id"].eq("")
        if missing_prescreen.any():
            out.loc[missing_prescreen, "prescreen_recipe_id"] = out.loc[missing_prescreen].apply(
                _prescreen_id_from_recipe_fields, axis=1
            )
    if "recipe_config_id" not in out.columns:
        out["recipe_config_id"] = out.apply(recipe_config_id, axis=1)
    if "inner_selection_rank" not in out.columns:
        out["inner_selection_rank"] = out.groupby("outer_fold").cumcount().astype(int) + 1
    return out.reset_index(drop=True)


def _normalise_stage02_score_frame(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="coerce")
    out = out.loc[out["outer_fold"].notna()].copy()
    out["outer_fold"] = out["outer_fold"].astype(int)
    out["feature_size"] = pd.to_numeric(out["feature_size"], errors="coerce").astype(int)
    if "prescreen_recipe_id" not in out.columns:
        out["prescreen_recipe_id"] = out.apply(_prescreen_id_from_recipe_fields, axis=1)
    if "recipe_config_id" not in out.columns:
        out["recipe_config_id"] = out.apply(recipe_config_id, axis=1)
    return out


def select_global_stage02_pipeline_recipes(
    stage_two_dir: Path,
    *,
    outer_folds: Sequence[int],
    top_n: int = 20,
    min_inner_fold_count: int = 5,
    max_feature_size: int | None = None,
) -> pd.DataFrame:
    """Select global stage-03 candidate configs from notebook-02 round-2 scores.

    Notebook 02 selects and scores recipes separately inside each outer-train block.
    Notebook 03 must audit final candidate *configurations* on every outer fold to avoid
    survivorship bias. This helper ranks global ``recipe_config_id`` values from the
    stage-02 round-2 score table, then expands each selected config across all requested
    outer folds.
    """
    stage_two_dir = Path(stage_two_dir)
    score_path = stage_two_dir / "round2_inner_oof_scores.csv"
    if not score_path.exists() or score_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing stage-02 round-2 OOF score artifact: {score_path}")

    scores = _normalise_stage02_score_frame(pd.read_csv(score_path))
    if "status" in scores.columns:
        scores = scores.loc[scores["status"].eq("ok")].copy()
    if max_feature_size is not None:
        scores = scores.loc[scores["feature_size"].le(int(max_feature_size))].copy()
    if "inner_fold_count" in scores.columns:
        scores = scores.loc[
            pd.to_numeric(scores["inner_fold_count"], errors="coerce").ge(int(min_inner_fold_count))
        ].copy()
    for completeness_col in ("missing_inner_oof_predictions", "missing_inner_fold_count"):
        if completeness_col in scores.columns:
            scores = scores.loc[
                pd.to_numeric(scores[completeness_col], errors="coerce").fillna(0).eq(0)
            ].copy()
    if scores.empty:
        return pd.DataFrame()

    first_cols = [
        "pipeline_recipe_id",
        "feature_recipe_id",
        "prescreen_recipe_id",
        "prescreen_name",
        "prescreen_methods",
        "rank_aggregation",
        "feature_size",
        "model_spec_id",
        "base_model_family",
        "model_kind",
        "model_params",
        "selected_features_text",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "first") for column in first_cols if column in scores.columns
    }
    metric_aggs: dict[str, tuple[str, str]] = {
        "stage02_outer_fold_count": ("outer_fold", "nunique"),
        "inner_oof_business_score": ("inner_oof_business_score", "mean"),
        "median_inner_oof_business_score": ("inner_oof_business_score", "median"),
        "min_inner_oof_business_score": ("inner_oof_business_score", "min"),
        "max_inner_oof_business_score": ("inner_oof_business_score", "max"),
        "inner_oof_optimal_k": ("inner_oof_optimal_k", "mean"),
        "inner_oof_f1_score": ("inner_oof_f1_score", "mean"),
        "inner_oof_roc_auc_score": ("inner_oof_roc_auc_score", "mean"),
    }
    for output_col, spec in metric_aggs.items():
        source_col, _func = spec
        if source_col in scores.columns:
            aggregations[output_col] = spec

    global_configs = (
        scores.groupby("recipe_config_id", as_index=False)
        .agg(**aggregations)
        .sort_values(
            [
                "inner_oof_business_score",
                "min_inner_oof_business_score",
                "stage02_outer_fold_count",
                "feature_size",
                "inner_oof_roc_auc_score",
            ],
            ascending=[False, False, False, True, False],
            na_position="last",
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
    global_configs["global_candidate_rank"] = range(1, len(global_configs) + 1)
    global_configs["inner_selection_rank"] = global_configs["global_candidate_rank"]
    global_configs["selection_artifact"] = score_path.name

    frames: list[pd.DataFrame] = []
    for outer_fold in sorted({int(fold) for fold in outer_folds}):
        fold_frame = global_configs.copy()
        fold_frame["outer_fold"] = int(outer_fold)
        frames.append(fold_frame)
    if not frames:
        return pd.DataFrame()
    return normalize_pipeline_recipes(pd.concat(frames, ignore_index=True))


def complete_outer_prediction_pairs(
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    *,
    outer_folds: Sequence[int] | None = None,
) -> set[tuple[int, str]]:
    """Return complete ``(outer_fold, recipe_config_id)`` pairs in saved stage-03 outputs."""
    if scores.empty or predictions.empty:
        return set()
    required_scores = {"outer_fold", "recipe_config_id", "status"}
    required_predictions = {"outer_fold", "recipe_config_id", "sample_index", "status"}
    if not required_scores.issubset(scores.columns) or not required_predictions.issubset(
        predictions.columns
    ):
        return set()

    allowed_folds = {int(fold) for fold in outer_folds} if outer_folds is not None else None
    assignment_rows = outer_assignments.copy()
    if allowed_folds is not None:
        assignment_rows = assignment_rows.loc[assignment_rows["outer_fold"].isin(allowed_folds)]
    expected_samples = {
        int(outer_fold): set(group["sample_index"].map(_python_scalar).tolist())
        for outer_fold, group in assignment_rows.groupby("outer_fold")
    }

    ok_score_pairs = {
        (int(row.outer_fold), str(row.recipe_config_id))
        for row in scores.loc[scores["status"].eq("ok"), ["outer_fold", "recipe_config_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    complete_pairs: set[tuple[int, str]] = set()
    ok_predictions = predictions.loc[predictions["status"].eq("ok")].copy()
    for (outer_fold, config_id), group in ok_predictions.groupby(
        ["outer_fold", "recipe_config_id"], sort=False
    ):
        outer_fold = int(outer_fold)
        config_id = str(config_id)
        if allowed_folds is not None and outer_fold not in allowed_folds:
            continue
        if (outer_fold, config_id) not in ok_score_pairs:
            continue
        expected = expected_samples.get(outer_fold)
        if expected is None:
            continue
        observed = set(group["sample_index"].map(_python_scalar).tolist())
        if observed == expected and len(group) == len(expected):
            complete_pairs.add((outer_fold, config_id))
    return complete_pairs


def filter_outer_recipe_pairs(frame: pd.DataFrame, pairs: set[tuple[int, str]]) -> pd.DataFrame:
    """Keep rows whose ``(outer_fold, recipe_config_id)`` pair is in ``pairs``."""
    if frame.empty or not pairs or not {"outer_fold", "recipe_config_id"}.issubset(frame.columns):
        return frame.head(0).copy()
    mask = frame.apply(
        lambda row: (int(row["outer_fold"]), str(row["recipe_config_id"])) in pairs,
        axis=1,
    )
    return frame.loc[mask].copy()


def load_or_select_stage02_pipeline_recipes(
    stage_two_dir: Path,
    *,
    outer_folds: Sequence[int] | None = None,
    inner_assignments: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    top_n: int = 10,
    min_inner_fold_count: int = 5,
    max_targets: int = DEFAULT_MAX_TARGETS,
    chunksize: int = 250_000,
    random_state: int = 42,
    rebuild_from_predictions: bool = True,
    write_outputs: bool = True,
    prefer_selected_artifact: bool = True,
    max_feature_size: int | None = None,
) -> pd.DataFrame:
    """Load notebook-02 selected recipes or reconstruct them from notebook-02 outputs."""
    stage_two_dir = Path(stage_two_dir)
    selected_path = stage_two_dir / "outer_selected_pipeline_recipes.csv"
    if prefer_selected_artifact and selected_path.exists() and selected_path.stat().st_size > 0:
        selected = pd.read_csv(selected_path)
        selected["selection_artifact"] = selected_path.name
        selected = normalize_pipeline_recipes(selected)
        if max_feature_size is not None and "feature_size" in selected.columns:
            selected = selected.loc[selected["feature_size"].le(int(max_feature_size))].copy()
        return selected.reset_index(drop=True)

    oof_score_path = stage_two_dir / "round2_inner_oof_scores.csv"
    legacy_path = stage_two_dir / "round2_inner_scores.csv"
    prediction_path = stage_two_dir / "round2_inner_predictions.csv"

    if oof_score_path.exists() and oof_score_path.stat().st_size > 0:
        scores = pd.read_csv(oof_score_path)
        selection_artifact = oof_score_path.name
    elif legacy_path.exists() and legacy_path.stat().st_size > 0:
        scores = pd.read_csv(legacy_path, low_memory=False)
        selection_artifact = legacy_path.name
    elif rebuild_from_predictions and has_inner_prediction_columns(prediction_path):
        score_inner_oof_predictions_csv(
            prediction_path,
            max_targets=max_targets,
            chunksize=chunksize,
            random_repeats=0,
            random_state=random_state,
            outer_folds=outer_folds,
            round_names=("round2_model_hpo_selection",),
            output_path=oof_score_path if write_outputs else None,
            baseline_output_path=stage_two_dir / "round2_inner_oof_baselines.csv"
            if write_outputs
            else None,
            overwrite=True,
            inner_assignments=inner_assignments,
            y=y,
            verbose=True,
        )
        scores = pd.read_csv(oof_score_path)
        selection_artifact = oof_score_path.name
    else:
        raise FileNotFoundError("No stage-02 selected recipes or round-2 score artifacts found")

    scores = scores.copy()
    scores["outer_fold"] = pd.to_numeric(scores["outer_fold"], errors="coerce")
    scores = scores.loc[scores["outer_fold"].notna()].copy()
    scores["outer_fold"] = scores["outer_fold"].astype(int)
    if max_feature_size is not None and "feature_size" in scores.columns:
        scores = scores.loc[
            pd.to_numeric(scores["feature_size"], errors="coerce").le(int(max_feature_size))
        ].copy()
    effective_outer_folds = (
        sorted({int(fold) for fold in outer_folds})
        if outer_folds is not None
        else sorted(scores["outer_fold"].drop_duplicates())
    )

    frames: list[pd.DataFrame] = []
    for outer_fold in effective_outer_folds:
        selected = select_top_pipeline_recipes(
            scores.loc[scores["outer_fold"].eq(int(outer_fold))].copy(),
            top_n=top_n,
            min_inner_fold_count=min_inner_fold_count,
        )
        if not selected.empty:
            selected["outer_fold"] = int(outer_fold)
            selected["selection_artifact"] = selection_artifact
            frames.append(selected)
    if not frames:
        return pd.DataFrame()
    selected = normalize_pipeline_recipes(pd.concat(frames, ignore_index=True))
    if write_outputs:
        selected.to_csv(selected_path, index=False)
    return selected


def score_topk_predictions(
    y_true: Sequence[int] | pd.Series,
    scores: Sequence[float] | np.ndarray,
    *,
    feature_count: int,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> dict[str, Any]:
    """Score one prediction vector with the final top-k objective."""
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    curve = topk_business_curve(
        y_array, score_array, feature_count=feature_count, max_targets=max_targets
    )
    best = curve.iloc[int(curve["business_score"].idxmax())]
    try:
        f1 = build_f1_curve(pd.Series(y_array), score_array, max_targets=max_targets)
        f1_score = f1.best_f1
        roc_auc_score = f1.roc_auc
    except ValueError:
        f1_score = np.nan
        roc_auc_score = np.nan
    return {
        "outer_business_score": float(best["business_score"]),
        "outer_optimal_k": int(best["k"]),
        "outer_tp_at_optimal_k": int(best["tp"]),
        "outer_fp_at_optimal_k": int(best["fp"]),
        "outer_gross_score": float(best["gross_score"]),
        "feature_penalty": float(best["feature_penalty"]),
        "outer_f1_score": f1_score,
        "outer_roc_auc_score": roc_auc_score,
        "outer_n_predictions": len(y_array),
        "outer_selected_rate": float(best["k"] / len(y_array)) if len(y_array) else 0.0,
        "outer_threshold": float(best["threshold"]) if pd.notna(best["threshold"]) else np.nan,
        "score_curve": curve,
    }


def _outer_score_metadata_lookup(
    original_scores: pd.DataFrame | None,
) -> dict[tuple[int, str], pd.Series]:
    if original_scores is None or original_scores.empty:
        return {}
    required = {"outer_fold", "recipe_config_id"}
    if not required.issubset(original_scores.columns):
        return {}
    lookup: dict[tuple[int, str], pd.Series] = {}
    for (outer_fold, config_id), group in original_scores.groupby(
        ["outer_fold", "recipe_config_id"], sort=False
    ):
        lookup[(int(outer_fold), str(config_id))] = group.iloc[0]
    return lookup


def rescore_outer_predictions_at_targets(
    predictions: pd.DataFrame,
    *,
    max_targets_by_outer_fold: Mapping[int, int] | int,
    gross_score_scale_by_outer_fold: Mapping[int, float] | float = 1.0,
    original_scores: pd.DataFrame | None = None,
    scoring_strategy: str = "rate_matched_top1000_test",
) -> pd.DataFrame:
    """Rescore saved outer predictions without refitting any model.

    Notebook 03 outer folds have about 1000 rows. The final submission selects top 1000
    from the full test set, so a rate-matched audit should cap each outer fold at roughly
    the same contact rate. When scoring only one outer fold, gross TP/FP value must then
    be scaled back to the full test population while feature cost is charged once.
    """
    if predictions.empty:
        return pd.DataFrame()
    required = {"outer_fold", "recipe_config_id", "y_true", "score", "feature_size"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"outer predictions missing required columns: {sorted(missing)}")

    ok_predictions = predictions.copy()
    if "status" in ok_predictions.columns:
        ok_predictions = ok_predictions.loc[ok_predictions["status"].eq("ok")].copy()
    if ok_predictions.empty:
        return pd.DataFrame()

    metadata_lookup = _outer_score_metadata_lookup(original_scores)
    sample_columns = {"sample_index", "y_true", "score", "status", "error"}
    metadata_columns = [column for column in ok_predictions.columns if column not in sample_columns]
    rows: list[dict[str, Any]] = []
    for (outer_fold, config_id), group in ok_predictions.groupby(
        ["outer_fold", "recipe_config_id"], sort=False
    ):
        outer_fold = int(outer_fold)
        config_id = str(config_id)
        metadata = group.iloc[0][metadata_columns].to_dict()
        original = metadata_lookup.get((outer_fold, config_id))
        if original is not None:
            for column, value in original.items():
                if column not in metadata or pd.isna(metadata.get(column)):
                    metadata[column] = value
            for column in ("selected_features_text", "selected_feature_count"):
                if column in original.index:
                    metadata[column] = original[column]
        metadata.setdefault("selected_features_text", "")
        metadata.setdefault("selected_feature_count", metadata.get("feature_size"))
        metadata.setdefault("inner_oof_business_score", np.nan)
        metadata.setdefault("inner_selection_rank", np.nan)
        if isinstance(max_targets_by_outer_fold, Mapping):
            max_targets = int(max_targets_by_outer_fold[outer_fold])
        else:
            max_targets = int(max_targets_by_outer_fold)
        if isinstance(gross_score_scale_by_outer_fold, Mapping):
            gross_score_scale = float(gross_score_scale_by_outer_fold[outer_fold])
        else:
            gross_score_scale = float(gross_score_scale_by_outer_fold)
        scored = score_topk_predictions(
            group["y_true"].astype(int).reset_index(drop=True),
            group["score"].astype(float).to_numpy(),
            feature_count=int(float(metadata["selected_feature_count"])),
            max_targets=max_targets,
        )
        raw_business_score = float(scored["outer_business_score"])
        raw_gross_score = float(scored["outer_gross_score"])
        feature_penalty = float(scored["feature_penalty"])
        scaled_gross_score = raw_gross_score * gross_score_scale
        scaled_business_score = scaled_gross_score - feature_penalty
        scored = dict(scored)
        scored["outer_business_score"] = scaled_business_score
        scored["outer_gross_score"] = scaled_gross_score
        rows.append({
            **metadata,
            "outer_fold": outer_fold,
            "recipe_config_id": config_id,
            "status": "ok",
            "error": "",
            "outer_scoring_strategy": scoring_strategy,
            "outer_max_targets": max_targets,
            "outer_gross_score_scale": gross_score_scale,
            "outer_business_score_unscaled_rate_fold": raw_business_score,
            "outer_gross_score_unscaled_rate_fold": raw_gross_score,
            **{key: value for key, value in scored.items() if key != "score_curve"},
        })
    return pd.DataFrame(rows)


def _model_spec_from_recipe(recipe: pd.Series, model_specs: pd.DataFrame | None) -> pd.Series:
    values = {
        "model_spec_id": recipe.get("model_spec_id", ""),
        "base_model_family": recipe.get("base_model_family", ""),
        "model_kind": recipe.get("model_kind", ""),
        "model_params": recipe.get("model_params", "{}"),
    }
    if model_specs is not None and not model_specs.empty and values["model_spec_id"]:
        matches = model_specs.loc[model_specs["model_spec_id"].eq(values["model_spec_id"])]
        if not matches.empty:
            lookup = matches.iloc[0]
            for column in ("base_model_family", "model_kind", "model_params"):
                if not values.get(column) or pd.isna(values[column]):
                    values[column] = lookup[column]
    return pd.Series(values)


def _selected_candidate_from_rankings(
    rankings: pd.DataFrame,
    prescreen_recipe: pd.Series,
    *,
    feature_size: int,
) -> pd.Series:
    prescreen_recipe = prescreen_recipe.copy()
    if "prescreen_recipe_id" not in prescreen_recipe.index and prescreen_recipe.name is not None:
        prescreen_recipe["prescreen_recipe_id"] = prescreen_recipe.name
    candidates, failures = build_fold_feature_candidates(
        rankings,
        pd.DataFrame([prescreen_recipe]),
        [int(feature_size)],
        max_feature_candidates=None,
        dedupe_feature_sets=False,
    )
    if candidates.empty:
        error = "no feature candidate materialized"
        if not failures.empty and "error" in failures:
            error = str(failures.iloc[0]["error"])
        raise ValueError(error)
    return candidates.iloc[0]


def _recipe_metadata(recipe: pd.Series, *, config_id: str) -> dict[str, Any]:
    fields = [
        "pipeline_recipe_id",
        "feature_recipe_id",
        "prescreen_recipe_id",
        "prescreen_name",
        "prescreen_methods",
        "rank_aggregation",
        "feature_size",
        "model_spec_id",
        "base_model_family",
        "model_kind",
        "model_params",
        "inner_selection_rank",
        "inner_oof_business_score",
        "inner_oof_optimal_k",
        "inner_oof_f1_score",
        "inner_oof_roc_auc_score",
        "selection_artifact",
        "outer_config_rank",
        "finalist_rank",
        "mean_outer_business_score",
        "median_outer_business_score",
        "outer_fold_count",
        "mean_outer_optimal_k",
    ]
    out = {"recipe_config_id": config_id}
    for field in fields:
        if field in recipe.index:
            out[field] = recipe[field]
    return out


def run_outer_fold_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    selected_recipes: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    prescreen_recipes: pd.DataFrame,
    model_specs: pd.DataFrame | None = None,
    *,
    outer_folds: Sequence[int] | None = None,
    random_state: int = 42,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit selected recipes from scratch on each outer train and score outer val."""
    selected = normalize_pipeline_recipes(selected_recipes)
    if outer_folds is not None:
        selected = selected.loc[
            selected["outer_fold"].isin({int(fold) for fold in outer_folds})
        ].copy()
    prescreen_lookup = prescreen_recipes.set_index("prescreen_recipe_id")
    score_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []

    for outer_fold in sorted(selected["outer_fold"].drop_duplicates()):
        fold_selected = selected.loc[selected["outer_fold"].eq(int(outer_fold))].copy()
        train_idx, val_idx = outer_fold_indices(outer_assignments, int(outer_fold))
        x_outer_train = X.iloc[list(train_idx)]
        y_outer_train = y.iloc[list(train_idx)]
        x_outer_val = X.iloc[list(val_idx)]
        y_outer_val = y.iloc[list(val_idx)]
        fold_prescreens = prescreen_recipes.loc[
            prescreen_recipes["prescreen_recipe_id"].isin(
                fold_selected["prescreen_recipe_id"].drop_duplicates()
            )
        ].copy()
        methods = required_prescreen_methods(fold_prescreens)
        print(
            f"[outer={outer_fold}] fitting {len(methods)} prescreen methods "
            f"on {len(train_idx)} outer-train rows"
        )
        rankings, method_failures = collect_prescreen_rankings(
            x_outer_train,
            y_outer_train,
            methods=methods,
            random_state=random_state + int(outer_fold) * 10_000,
            continue_on_error=True,
        )
        if not method_failures.empty:
            print(f"[outer={outer_fold}] prescreen method failures={len(method_failures)}")

        for _, recipe in fold_selected.iterrows():
            started = time.perf_counter()
            config_id = str(recipe.get("recipe_config_id") or recipe_config_id(recipe))
            try:
                candidate = _selected_candidate_from_rankings(
                    rankings,
                    prescreen_lookup.loc[recipe["prescreen_recipe_id"]],
                    feature_size=int(recipe["feature_size"]),
                )
                features = parse_feature_names(candidate["selected_features"])
                if not features:
                    raise ValueError("empty selected feature list")
                model_spec = _model_spec_from_recipe(recipe, model_specs)
                scores = fit_predict_model_spec(
                    x_outer_train,
                    y_outer_train,
                    x_outer_val,
                    features,
                    model_spec,
                    random_state=random_state + int(outer_fold) * 10_000,
                )
                scored = score_topk_predictions(
                    y_outer_val,
                    scores,
                    feature_count=len(features),
                    max_targets=max_targets,
                )
                selected_features_text = ",".join(features)
                score_rows.append({
                    **_recipe_metadata(recipe, config_id=config_id),
                    "outer_fold": int(outer_fold),
                    "selected_features_text": selected_features_text,
                    "selected_feature_count": len(features),
                    "status": "ok",
                    "error": "",
                    "duration_seconds": time.perf_counter() - started,
                    **{key: value for key, value in scored.items() if key != "score_curve"},
                })
                prediction_frames.append(
                    pd.DataFrame({
                        **_recipe_metadata(recipe, config_id=config_id),
                        "outer_fold": int(outer_fold),
                        "sample_index": list(val_idx),
                        "y_true": y_outer_val.to_numpy(dtype=int),
                        "score": np.asarray(scores, dtype=float),
                        "status": "ok",
                        "error": "",
                    })
                )
                for feature_order, feature in enumerate(features, start=1):
                    feature_rows.append({
                        **_recipe_metadata(recipe, config_id=config_id),
                        "outer_fold": int(outer_fold),
                        "feature_order": feature_order,
                        "feature": feature,
                        "selected_features_text": selected_features_text,
                        "status": "ok",
                    })
                print(
                    f"[outer={outer_fold}] {config_id} "
                    f"score={scored['outer_business_score']:.1f} "
                    f"k={scored['outer_optimal_k']}"
                )
            except Exception as exc:
                score_rows.append({
                    **_recipe_metadata(recipe, config_id=config_id),
                    "outer_fold": int(outer_fold),
                    "selected_features_text": "",
                    "selected_feature_count": np.nan,
                    "status": "error",
                    "error": repr(exc),
                    "duration_seconds": time.perf_counter() - started,
                })
                print(f"[outer={outer_fold}] {config_id} failed: {exc!r}")

    scores = pd.DataFrame(score_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    selected_features = pd.DataFrame(feature_rows)
    if not scores.empty:
        sort_cols = [
            column
            for column in ("outer_fold", "status", "outer_business_score", "inner_selection_rank")
            if column in scores.columns
        ]
        ascending = [
            column in {"outer_fold", "status", "inner_selection_rank"} for column in sort_cols
        ]
        scores = scores.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(
            drop=True
        )
    return scores, predictions, selected_features


def aggregate_outer_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate honest outer scores by recipe/model/feature-size config."""
    if scores.empty:
        return pd.DataFrame()
    ok_scores = scores.loc[scores["status"].eq("ok")].copy()
    if ok_scores.empty:
        return pd.DataFrame()
    group_cols = [
        "recipe_config_id",
        "prescreen_recipe_id",
        "prescreen_name",
        "feature_size",
        "model_spec_id",
        "base_model_family",
        "model_kind",
        "model_params",
    ]
    grouped = (
        ok_scores.groupby(group_cols, as_index=False)
        .agg(
            outer_fold_count=("outer_fold", "nunique"),
            mean_outer_business_score=("outer_business_score", "mean"),
            median_outer_business_score=("outer_business_score", "median"),
            min_outer_business_score=("outer_business_score", "min"),
            max_outer_business_score=("outer_business_score", "max"),
            std_outer_business_score=("outer_business_score", "std"),
            mean_outer_optimal_k=("outer_optimal_k", "mean"),
            median_outer_optimal_k=("outer_optimal_k", "median"),
            mean_outer_f1_score=("outer_f1_score", "mean"),
            mean_outer_roc_auc_score=("outer_roc_auc_score", "mean"),
            mean_inner_oof_business_score=("inner_oof_business_score", "mean"),
            mean_inner_selection_rank=("inner_selection_rank", "mean"),
            selected_feature_count=("selected_feature_count", "median"),
            example_selected_features_text=("selected_features_text", "first"),
        )
        .sort_values(
            [
                "mean_outer_business_score",
                "median_outer_business_score",
                "outer_fold_count",
                "std_outer_business_score",
                "feature_size",
            ],
            ascending=[False, False, False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    grouped["outer_config_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def select_stable_configurations(
    outer_summary: pd.DataFrame,
    *,
    top_n: int = 10,
    min_outer_folds: int = 2,
) -> pd.DataFrame:
    """Select final candidate configs using only honest outer-evaluation results."""
    if outer_summary.empty:
        return outer_summary.copy()
    stable = outer_summary.loc[outer_summary["outer_fold_count"].ge(int(min_outer_folds))].copy()
    if stable.empty:
        stable = outer_summary.copy()
        stable["stability_note"] = (
            f"fallback: no config appeared in at least {int(min_outer_folds)} outer folds"
        )
    else:
        stable["stability_note"] = f"outer_fold_count >= {int(min_outer_folds)}"
    stable = stable.head(int(top_n)).copy()
    stable["finalist_rank"] = np.arange(1, len(stable) + 1)
    return stable.reset_index(drop=True)


def select_cost_aware_configurations(
    outer_summary: pd.DataFrame,
    *,
    top_n: int = 10,
    min_outer_folds: int = 2,
    max_feature_size: int | None = None,
    fallback_min_outer_folds: int = 1,
    require_exact_outer_folds: bool = False,
    allow_incomplete_fallback: bool = True,
) -> pd.DataFrame:
    """Select lower-cost finalists from an honest outer summary.

    This keeps the selection based only on notebook-03 outer scores, but applies a hard
    feature-count cap first. When ``allow_incomplete_fallback`` is false, no config with
    insufficient outer-fold coverage can be selected.
    """
    if outer_summary.empty:
        return outer_summary.copy()

    def ordered(frame: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [
            "mean_outer_business_score",
            "min_outer_business_score",
            "outer_fold_count",
            "feature_size",
        ]
        sort_cols = [column for column in sort_cols if column in frame.columns]
        ascending = [False, False, False, True][: len(sort_cols)]
        return frame.sort_values(sort_cols, ascending=ascending, na_position="last")

    candidates = outer_summary.copy()
    selected_frames: list[pd.DataFrame] = []
    used_ids: set[str] = set()

    if max_feature_size is not None and "feature_size" in candidates.columns:
        low_cost = candidates.loc[candidates["feature_size"].le(int(max_feature_size))].copy()
    else:
        low_cost = candidates.copy()

    if require_exact_outer_folds:
        primary_mask = low_cost["outer_fold_count"].eq(int(min_outer_folds))
        fold_count_note = f"outer_fold_count == {int(min_outer_folds)}"
    else:
        primary_mask = low_cost["outer_fold_count"].ge(int(min_outer_folds))
        fold_count_note = f"outer_fold_count >= {int(min_outer_folds)}"
    primary = low_cost.loc[primary_mask].copy()
    if not primary.empty:
        primary = ordered(primary).head(int(top_n)).copy()
        primary["selection_note"] = (
            f"feature_size <= {max_feature_size}; {fold_count_note}"
            if max_feature_size is not None
            else fold_count_note
        )
        selected_frames.append(primary)
        used_ids.update(primary["recipe_config_id"].astype(str))

    if allow_incomplete_fallback and sum(len(frame) for frame in selected_frames) < int(top_n):
        fallback = low_cost.loc[
            low_cost["outer_fold_count"].ge(int(fallback_min_outer_folds))
            & ~low_cost["recipe_config_id"].astype(str).isin(used_ids)
        ].copy()
        if not fallback.empty:
            remaining = int(top_n) - sum(len(frame) for frame in selected_frames)
            fallback = ordered(fallback).head(remaining).copy()
            fallback["selection_note"] = (
                f"cost fallback: feature_size <= {max_feature_size}; "
                f"outer_fold_count >= {int(fallback_min_outer_folds)}"
                if max_feature_size is not None
                else f"fallback outer_fold_count >= {int(fallback_min_outer_folds)}"
            )
            selected_frames.append(fallback)
            used_ids.update(fallback["recipe_config_id"].astype(str))

    if allow_incomplete_fallback and sum(len(frame) for frame in selected_frames) < int(top_n):
        fallback = candidates.loc[~candidates["recipe_config_id"].astype(str).isin(used_ids)].copy()
        if not fallback.empty:
            remaining = int(top_n) - sum(len(frame) for frame in selected_frames)
            fallback = ordered(fallback).head(remaining).copy()
            fallback["selection_note"] = "final fallback: cost cap relaxed"
            selected_frames.append(fallback)

    if not selected_frames:
        empty = outer_summary.head(0).copy()
        for column in ("selection_note", "stability_note", "finalist_rank"):
            if column not in empty.columns:
                empty[column] = pd.Series(dtype="object")
        return empty
    selected = pd.concat(selected_frames, ignore_index=True).head(int(top_n)).copy()
    selected["stability_note"] = selected.get("selection_note", "cost-aware selection")
    selected["finalist_rank"] = np.arange(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def refit_final_configurations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    finalists: pd.DataFrame,
    prescreen_recipes: pd.DataFrame,
    model_specs: pd.DataFrame | None = None,
    *,
    random_state: int = 42,
    top_n: int = DEFAULT_MAX_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit finalist configs on all train and rank all test rows."""
    if finalists.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prescreen_lookup = prescreen_recipes.set_index("prescreen_recipe_id")
    needed_prescreens = prescreen_recipes.loc[
        prescreen_recipes["prescreen_recipe_id"].isin(finalists["prescreen_recipe_id"])
    ]
    methods = required_prescreen_methods(needed_prescreens)
    print(f"[final] fitting {len(methods)} prescreen methods on all {len(X_train)} train rows")
    rankings, method_failures = collect_prescreen_rankings(
        X_train, y_train, methods=methods, random_state=random_state, continue_on_error=True
    )
    if not method_failures.empty:
        print(f"[final] prescreen method failures={len(method_failures)}")

    manifest_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    for _, finalist in finalists.iterrows():
        started = time.perf_counter()
        config_id = str(finalist.get("recipe_config_id") or recipe_config_id(finalist))
        finalist_rank = int(finalist.get("finalist_rank", len(manifest_rows) + 1))
        try:
            candidate = _selected_candidate_from_rankings(
                rankings,
                prescreen_lookup.loc[finalist["prescreen_recipe_id"]],
                feature_size=int(finalist["feature_size"]),
            )
            features = parse_feature_names(candidate["selected_features"])
            if not features:
                raise ValueError("empty selected feature list")
            model_spec = _model_spec_from_recipe(finalist, model_specs)
            test_scores = fit_predict_model_spec(
                X_train, y_train, X_test, features, model_spec, random_state=random_state
            )
            scores = np.asarray(test_scores, dtype=float)
            order = np.argsort(scores)[::-1]
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            selected_features_text = ",".join(features)
            manifest_rows.append({
                **_recipe_metadata(finalist, config_id=config_id),
                "finalist_rank": finalist_rank,
                "selected_features_text": selected_features_text,
                "selected_feature_count": len(features),
                "test_rows_scored": len(X_test),
                "top_n_saved": min(int(top_n), len(X_test)),
                "status": "ok",
                "error": "",
                "duration_seconds": time.perf_counter() - started,
            })
            prediction_frames.append(
                pd.DataFrame({
                    **_recipe_metadata(finalist, config_id=config_id),
                    "finalist_rank": finalist_rank,
                    "sample_index": np.arange(len(scores)),
                    "score": scores,
                    "rank": ranks,
                    "selected_for_top_n": ranks <= min(int(top_n), len(scores)),
                    "status": "ok",
                })
            )
            for feature_order, feature in enumerate(features, start=1):
                feature_rows.append({
                    **_recipe_metadata(finalist, config_id=config_id),
                    "finalist_rank": finalist_rank,
                    "feature_order": feature_order,
                    "feature": feature,
                    "selected_features_text": selected_features_text,
                    "status": "ok",
                })
            print(f"[final] {config_id} ranked {len(scores)} test rows")
        except Exception as exc:
            manifest_rows.append({
                **_recipe_metadata(finalist, config_id=config_id),
                "finalist_rank": finalist_rank,
                "selected_features_text": "",
                "selected_feature_count": np.nan,
                "test_rows_scored": 0,
                "top_n_saved": 0,
                "status": "error",
                "error": repr(exc),
                "duration_seconds": time.perf_counter() - started,
            })
            print(f"[final] {config_id} failed: {exc!r}")
    manifest = pd.DataFrame(manifest_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    features = pd.DataFrame(feature_rows)
    return manifest, predictions, features


def build_rank_mean_ensemble(
    predictions: pd.DataFrame,
    finalist_manifest: pd.DataFrame,
    *,
    top_n: int = DEFAULT_MAX_TARGETS,
    ensemble_name: str = "rank_mean_ensemble",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a simple rank-mean test ensemble from notebook-03 finalists."""
    ok_manifest = finalist_manifest.loc[finalist_manifest["status"].eq("ok")].copy()
    if predictions.empty or len(ok_manifest) < 2:
        return pd.DataFrame(), pd.DataFrame()
    member_ids = ok_manifest["recipe_config_id"].tolist()
    wide = (
        predictions.loc[predictions["recipe_config_id"].isin(member_ids)]
        .pivot(index="sample_index", columns="recipe_config_id", values="score")
        .sort_index()
        .dropna(axis=1, how="any")
    )
    if wide.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()
    ensemble_score = wide.rank(axis=0, method="average", pct=True).mean(axis=1).to_numpy()
    order = np.argsort(ensemble_score)[::-1]
    top = order[: min(int(top_n), len(order))]
    ranking = pd.DataFrame({
        "rank": np.arange(1, len(top) + 1),
        "sample_index": wide.index.to_numpy()[top],
        "score": ensemble_score[top],
        "ensemble_name": ensemble_name,
        "member_count": wide.shape[1],
    })
    manifest = pd.DataFrame([
        {
            "ensemble_name": ensemble_name,
            "member_count": wide.shape[1],
            "member_recipe_config_ids": ",".join(wide.columns),
            "top_n_saved": len(ranking),
            "selection_rule": "members selected by honest outer evaluation in notebook 03",
        }
    ])
    return manifest, ranking
