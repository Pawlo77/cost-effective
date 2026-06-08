"""Model-family OOF validation helpers for sparse feature-set search."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cost_effective.dataset.ensemble_feature_selection import (
    rank_test_customers,
    score_oof_predictions,
)
from cost_effective.dataset.utils import DEFAULT_MAX_TARGETS
from cost_effective.models.model_specs import (
    EstimatorFactory,
    ModelFamilySpec,
    classifier_oof_scores,
    default_model_specs,
    ebm_factory,
    extra_trees_factory,
    fit_classifier_full,
    fit_lambdamart_full,
    lambdamart_oof_scores,
    lightgbm_factory,
    logistic_factory,
    make_classifier_spec,
    make_lambdamart_spec,
    positive_scores,
    random_forest_factory,
    scale_pos_weight,
    small_hpo_grid,
    specs_from_hpo_grid,
    weighted_scale_pos_weight,
    xgboost_factory,
)
from cost_effective.models.parsing import parse_feature_list

__all__ = [
    "EstimatorFactory",
    "ModelFamilySpec",
    "candidate_feature_sets_from_rankings",
    "classifier_oof_scores",
    "default_model_specs",
    "ebm_factory",
    "evaluate_model_family_specs",
    "extra_trees_factory",
    "feature_set_key",
    "fit_classifier_full",
    "fit_lambdamart_full",
    "lambdamart_oof_scores",
    "lightgbm_factory",
    "logistic_factory",
    "make_classifier_spec",
    "make_lambdamart_spec",
    "merge_candidate_feature_sets",
    "parse_feature_list",
    "positive_scores",
    "random_forest_factory",
    "rank_validation_scores",
    "scale_pos_weight",
    "select_unique_feature_sets",
    "small_hpo_grid",
    "specs_from_hpo_grid",
    "summarize_validation_scores",
    "weighted_scale_pos_weight",
    "xgboost_factory",
]


def feature_set_key(features: Iterable[str]) -> tuple[str, ...]:
    """Stable feature-set key independent of input ordering."""
    return tuple(sorted(dict.fromkeys(str(feature) for feature in features)))


def select_unique_feature_sets(
    experiment_results: pd.DataFrame,
    available_features: Sequence[str],
    *,
    top_n: int = 50,
    max_features: int = 60,
) -> pd.DataFrame:
    """Select top unique feature sets from previous experiment rows."""
    available = set(available_features)
    required = {"status", "source_features", "comparison_rank"}
    missing = required - set(experiment_results.columns)
    if missing:
        raise ValueError(f"experiment_results is missing columns: {sorted(missing)}")

    ok_results = experiment_results.loc[
        experiment_results["status"].eq("ok")
        & experiment_results["source_features"].notna()
        & experiment_results["comparison_rank"].notna()
    ].copy()
    ok_results = ok_results.sort_values(["comparison_rank", "source_feature_count"])

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for _, row in ok_results.iterrows():
        features = [
            feature
            for feature in parse_feature_list(row["source_features"])
            if feature in available
        ]
        key = feature_set_key(features)
        if not key or len(key) > max_features or key in seen:
            continue
        seen.add(key)
        rows.append({
            "feature_set_id": f"fs_{len(rows) + 1:03d}",
            "source_feature_count": len(key),
            "source_features": list(key),
            "source_features_text": ",".join(key),
            "origin_experiment_stage": row.get("experiment_stage", ""),
            "origin_method_family": row.get("method_family", ""),
            "origin_method": row.get("method", ""),
            "origin_oof_business_score": row.get("oof_business_score", np.nan),
            "origin_comparison_rank": row.get("comparison_rank", np.nan),
            "candidate_source": "previous_experiment",
        })
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def candidate_feature_sets_from_rankings(
    ranking_map: Mapping[str, pd.DataFrame],
    *,
    sizes: Sequence[int],
    max_features: int = 60,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """Generate deduplicated top-k feature-set candidates from prescreen rankings."""
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, ...], int] = {}
    for ranking_name, ranking in ranking_map.items():
        ordered = ranking.sort_values("order")["feature"].tolist()
        for size in sorted({size for size in sizes if size > 0}):
            if size > len(ordered) or size > max_features:
                continue
            features = ordered[:size]
            key = feature_set_key(features)
            if not key:
                continue
            score_col = "ensemble_score" if "ensemble_score" in ranking else None
            heuristic = (
                float(ranking.set_index("feature").loc[list(key), score_col].mean())
                if score_col
                else np.nan
            )
            if key in seen:
                idx = seen[key]
                if np.isfinite(heuristic) and heuristic > rows[idx].get(
                    "candidate_heuristic_score", -np.inf
                ):
                    rows[idx].update({
                        "origin_method": f"{ranking_name}::top_{size:02d}",
                        "origin_method_family": "prescreen_all_k",
                        "origin_experiment_stage": "prescreen_candidate_generation",
                        "origin_oof_business_score": np.nan,
                        "origin_comparison_rank": np.nan,
                        "candidate_heuristic_score": heuristic,
                    })
                continue
            seen[key] = len(rows)
            rows.append({
                "feature_set_id": "",
                "source_feature_count": len(key),
                "source_features": list(key),
                "source_features_text": ",".join(key),
                "origin_experiment_stage": "prescreen_candidate_generation",
                "origin_method_family": "prescreen_all_k",
                "origin_method": f"{ranking_name}::top_{size:02d}",
                "origin_oof_business_score": np.nan,
                "origin_comparison_rank": np.nan,
                "candidate_source": "all_k_prescreen",
                "candidate_heuristic_score": heuristic,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["candidate_heuristic_score", "source_feature_count", "origin_method"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if max_candidates is not None:
        out = out.head(max_candidates).copy()
    out["feature_set_id"] = [f"gen_{idx + 1:04d}" for idx in range(len(out))]
    return out


def merge_candidate_feature_sets(
    frames: Sequence[pd.DataFrame],
    *,
    top_n: int,
) -> pd.DataFrame:
    """Merge candidate feature-set frames by unique feature tuple."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            key = feature_set_key(row.get("source_features", []))
            if not key or key in seen:
                continue
            seen.add(key)
            item = row.to_dict()
            item["source_features"] = list(key)
            item["source_features_text"] = ",".join(key)
            rows.append(item)
            if len(rows) >= top_n:
                break
        if len(rows) >= top_n:
            break
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["feature_set_id"] = [f"fs_{idx + 1:03d}" for idx in range(len(out))]
    return out


def evaluate_model_family_specs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    specs: Sequence[ModelFamilySpec],
    *,
    output_dir: Path,
    project_root: Path,
    cv: int = 3,
    max_targets: int = DEFAULT_MAX_TARGETS,
    random_state: int = 42,
    save_test_rankings: bool = True,
) -> pd.DataFrame:
    """Evaluate feature sets under multiple model-family specs."""
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        subset = candidate_sets if spec.max_sets is None else candidate_sets.head(spec.max_sets)
        for _, feature_set in subset.iterrows():
            features = list(feature_set["source_features"])
            method_name = f"{spec.model_family}::{feature_set['feature_set_id']}"
            try:
                if spec.kind == "classifier":
                    if spec.factory is None:
                        raise ValueError(f"Missing classifier factory for {spec.model_family}")
                    oof = classifier_oof_scores(
                        X_train,
                        y_train,
                        features,
                        spec.factory,
                        cv=cv,
                        random_state=random_state,
                    )
                    full_model = fit_classifier_full(X_train, y_train, features, spec.factory)
                    test_scores = positive_scores(full_model, X_test[features])
                elif spec.kind == "lambdamart":
                    oof = lambdamart_oof_scores(
                        X_train,
                        y_train,
                        features,
                        params=spec.params,
                        cv=cv,
                        random_state=random_state,
                    )
                    full_model = fit_lambdamart_full(
                        X_train,
                        y_train,
                        features,
                        params=spec.params,
                        random_state=random_state,
                    )
                    test_scores = np.asarray(full_model.predict(X_test[features]), dtype=float)
                else:
                    raise ValueError(f"Unknown model kind: {spec.kind}")

                scored = score_oof_predictions(
                    y_train,
                    oof,
                    features,
                    method=method_name,
                    candidate_column_count=len(features),
                    max_targets=max_targets,
                )
                ranking_file = ""
                if save_test_rankings:
                    ranking_path = (
                        output_dir / f"{method_name.replace('::', '__')}_test_top1000.csv"
                    )
                    rank_test_customers(test_scores, k=max_targets).to_csv(
                        ranking_path, index=False
                    )
                    ranking_file = str(ranking_path.relative_to(project_root))
                rows.append(_validation_row(spec, feature_set, scored, "ok", "", ranking_file))
            except Exception as exc:
                rows.append(
                    _validation_row(
                        spec,
                        feature_set,
                        {
                            "method": method_name,
                            "source_features": features,
                            "source_feature_count": len(features),
                        },
                        "error",
                        repr(exc),
                        "",
                    )
                )
    return rank_validation_scores(pd.DataFrame(rows))


def rank_validation_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Attach validation_rank/status_order and sort validation rows."""
    if scores.empty:
        return scores
    scores = scores.copy()
    scores["validation_rank"] = np.nan
    ok_mask = scores["status"].eq("ok") & scores["validation_oof_business_score"].notna()
    ranked_idx = (
        scores.loc[ok_mask]
        .sort_values(
            ["validation_oof_business_score", "source_feature_count"],
            ascending=[False, True],
        )
        .index
    )
    scores.loc[ranked_idx, "validation_rank"] = np.arange(1, len(ranked_idx) + 1)
    scores["status_order"] = scores["status"].map({"ok": 0, "error": 1}).fillna(9).astype(int)
    return scores.sort_values(
        ["status_order", "validation_rank", "model_family", "feature_set_id"],
        na_position="last",
    ).reset_index(drop=True)


def summarize_validation_scores(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return best-by-model, best-by-feature-set, and stability summary tables."""
    ok_scores = scores.loc[scores["status"].eq("ok")].copy()
    if ok_scores.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    best_by_model = (
        ok_scores.sort_values(
            ["validation_oof_business_score", "source_feature_count"], ascending=[False, True]
        )
        .drop_duplicates("model_family")
        .reset_index(drop=True)
    )
    best_by_feature_set = (
        ok_scores.sort_values(
            ["validation_oof_business_score", "source_feature_count"], ascending=[False, True]
        )
        .drop_duplicates("feature_set_id")
        .reset_index(drop=True)
    )
    stability = (
        ok_scores.groupby("feature_set_id")
        .agg(
            n_model_families=("model_family", "nunique"),
            mean_business_score=("validation_oof_business_score", "mean"),
            median_business_score=("validation_oof_business_score", "median"),
            min_business_score=("validation_oof_business_score", "min"),
            max_business_score=("validation_oof_business_score", "max"),
            std_business_score=("validation_oof_business_score", "std"),
            best_validation_rank=("validation_rank", "min"),
            source_feature_count=("source_feature_count", "first"),
            source_features=("source_features", "first"),
            origin_method=("origin_method", "first"),
            origin_method_family=("origin_method_family", "first"),
            origin_oof_business_score=("origin_oof_business_score", "first"),
            origin_comparison_rank=("origin_comparison_rank", "first"),
        )
        .reset_index()
        .sort_values(["mean_business_score", "source_feature_count"], ascending=[False, True])
    )
    return best_by_model, best_by_feature_set, stability


def _validation_row(
    spec: ModelFamilySpec,
    feature_set: pd.Series,
    scored: Mapping[str, Any],
    status: str,
    error: str,
    ranking_file: str,
) -> dict[str, Any]:
    return {
        "model_family": spec.model_family,
        "model_kind": spec.kind,
        "model_params": dict(spec.params or {}),
        "feature_set_id": feature_set.get("feature_set_id", ""),
        "status": status,
        "error": error,
        "validation_method": scored.get("method", ""),
        "validation_oof_business_score": scored.get("oof_business_score", np.nan),
        "validation_oof_optimal_k": scored.get("oof_optimal_k", np.nan),
        "validation_f1_score": scored.get("f1_score", np.nan),
        "validation_roc_auc_score": scored.get("roc_auc_score", np.nan),
        "source_feature_count": scored.get(
            "source_feature_count", feature_set.get("source_feature_count", np.nan)
        ),
        "source_features": scored.get("source_features", feature_set.get("source_features", [])),
        "test_ranking_file": ranking_file,
        "origin_experiment_stage": feature_set.get("origin_experiment_stage", ""),
        "origin_method_family": feature_set.get("origin_method_family", ""),
        "origin_method": feature_set.get("origin_method", ""),
        "origin_oof_business_score": feature_set.get("origin_oof_business_score", np.nan),
        "origin_comparison_rank": feature_set.get("origin_comparison_rank", np.nan),
        "candidate_source": feature_set.get("candidate_source", ""),
        "candidate_heuristic_score": feature_set.get("candidate_heuristic_score", np.nan),
    }
